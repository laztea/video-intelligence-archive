# archive/web/app.py
import asyncio
import hashlib
from pathlib import Path
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse, HTMLResponse, FileResponse
from sse_starlette.sse import EventSourceResponse

from archive import config, embed
from archive.store.sqlite_store import SqliteStore
from archive.store.vector_store import VectorStore
from archive.search.engine import SearchEngine
from archive.search.query import decompose
from archive import pipeline
from archive.events import bus

app = FastAPI()
_store = None
_vector = None
STATIC = Path(__file__).parent / "static"

def get_store():
    global _store
    if _store is None:
        _store = SqliteStore(config.data_dir() / "app.db")
    return _store

def get_vector():
    global _vector
    if _vector is None:
        _vector = VectorStore(config.get_chroma())
    return _vector

def get_engine():
    return SearchEngine(get_store(), get_vector(), embed_fn=embed.embed_one)

def schedule_processing(video_id):
    asyncio.create_task(asyncio.to_thread(
        pipeline.process_video, video_id, get_store(), get_vector(), config.data_dir()))

@app.get("/", response_class=HTMLResponse)
def index():
    return (STATIC / "index.html").read_text(encoding="utf-8")

@app.get("/eval", response_class=HTMLResponse)
def eval_report():
    rp = Path(__file__).resolve().parent.parent.parent / "eval" / "report.html"
    if not rp.exists():
        return HTMLResponse(
            "<body style='font-family:system-ui;background:#14110d;color:#ece3d4;"
            "padding:60px;text-align:center'><h2>평가 리포트가 아직 없습니다</h2>"
            "<p><code>python eval/run_eval.py</code> 실행 후 새로고침하세요.</p></body>")
    return rp.read_text(encoding="utf-8")

@app.post("/upload")
async def upload(file: UploadFile = File(...)):
    data = await file.read()
    sha = hashlib.sha256(data).hexdigest()
    dest = config.data_dir() / "uploads" / f"{sha}{Path(file.filename).suffix}"
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_bytes(data)
    store = get_store()
    vid = store.insert_video(file.filename, str(dest), sha)
    schedule_processing(vid)
    return {"video_id": vid}

@app.get("/search")
def search(q: str = "", mode: str = "hybrid", kind: str = None, flagged: bool = False):
    filters = {}
    if kind: filters["kind"] = kind
    if flagged: filters["flagged"] = True
    if mode in ("hybrid", "vector") and q:
        parsed = decompose(q)
        semantic = parsed.get("semantic_query") or q
        # 키워드 검색은 GPT가 추출·확장한 키워드(유의어 포함)를 쓴다 (문장 X)
        keyword_query = " ".join(parsed.get("keywords") or [q])
        filters = {**parsed.get("filters", {}), **filters}
    else:
        semantic = q
        keyword_query = q
    results = get_engine().search(semantic, mode=mode, filters=filters,
                                  keyword_query=keyword_query)
    return JSONResponse(results)

@app.get("/videos")
def videos():
    store = get_store()
    out = []
    for v in store.list_videos():
        chunks = store.chunks_for_video(v["id"])
        flag_count = sum(len(store.flags_for_chunk(c["id"])) for c in chunks)
        out.append({**v, "chunk_count": len(chunks),
                    "scene_count": sum(1 for c in chunks if c["kind"] == "scene"),
                    "transcript_count": sum(1 for c in chunks if c["kind"] == "transcript"),
                    "flag_count": flag_count})
    return out

@app.get("/video/{video_id}")
def video_detail(video_id: int):
    store = get_store()
    v = store.get_video(video_id)
    if not v:
        raise HTTPException(404, "video not found")
    chunks = []
    for c in store.chunks_for_video(video_id):
        chunks.append({**c, "flags": store.flags_for_chunk(c["id"])})
    return {"video": v, "chunks": chunks}

@app.get("/media/{video_id}")
def media(video_id: int):
    v = get_store().get_video(video_id)
    if not v or not Path(v["stored_path"]).exists():
        raise HTTPException(404, "media not found")
    # FileResponse는 Range 요청을 지원해 브라우저 시킹이 가능하다.
    return FileResponse(v["stored_path"], media_type="video/mp4")

@app.get("/db/schema")
def db_schema():
    return get_store().schema()

@app.get("/db/table/{name}")
def db_table(name: str, limit: int = 100):
    try:
        return get_store().table_rows(name, limit)
    except ValueError:
        raise HTTPException(404, "unknown table")

@app.get("/thumb/{video_id}")
def thumb(video_id: int):
    frames_dir = config.data_dir() / "frames" / str(video_id)
    frames = sorted(frames_dir.glob("*.jpg")) if frames_dir.exists() else []
    if not frames:
        raise HTTPException(404, "no thumbnail")
    return FileResponse(frames[0], media_type="image/jpeg")

@app.get("/frame/{chunk_id}")
def chunk_frame(chunk_id: int):
    """검색 결과 썸네일: scene 청크는 자기 keyframe, 그 외엔 영상 첫 프레임."""
    store = get_store()
    c = store.get_chunk(chunk_id)
    if not c:
        raise HTTPException(404, "chunk not found")
    fp = c.get("frame_path")
    if fp and Path(fp).exists():
        return FileResponse(fp, media_type="image/jpeg")
    frames_dir = config.data_dir() / "frames" / str(c["video_id"])
    frames = sorted(frames_dir.glob("*.jpg")) if frames_dir.exists() else []
    if frames:
        return FileResponse(frames[0], media_type="image/jpeg")
    raise HTTPException(404, "no frame")

@app.get("/progress/{video_id}")
async def progress(video_id: int):
    queue = asyncio.Queue()
    def on_event(e):
        if e.video_id == video_id:
            queue.put_nowait({"step": e.step, "status": e.status, "message": e.message})
    bus.subscribe(on_event)
    async def gen():
        while True:
            ev = await queue.get()
            yield {"data": __import__("json").dumps(ev)}
            if ev["step"] == "finalize" and ev["status"] == "done":
                break
    return EventSourceResponse(gen())

@app.on_event("startup")
def _resume():
    if config.telegram_token():
        from archive.notify.telegram import subscribe_to_events, poll_loop
        subscribe_to_events()
        asyncio.create_task(poll_loop(get_engine()))
    asyncio.create_task(asyncio.to_thread(
        pipeline.resume_incomplete, get_store(), get_vector(), config.data_dir()))
