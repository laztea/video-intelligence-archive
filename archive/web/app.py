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
    parsed = decompose(q) if mode in ("hybrid", "vector") and q else {
        "semantic_query": q, "keywords": [q], "filters": {}}
    filters = {**parsed.get("filters", {}), **filters}
    results = get_engine().search(parsed["semantic_query"], mode=mode, filters=filters)
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

@app.get("/thumb/{video_id}")
def thumb(video_id: int):
    frames_dir = config.data_dir() / "frames" / str(video_id)
    frames = sorted(frames_dir.glob("*.jpg")) if frames_dir.exists() else []
    if not frames:
        raise HTTPException(404, "no thumbnail")
    return FileResponse(frames[0], media_type="image/jpeg")

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
