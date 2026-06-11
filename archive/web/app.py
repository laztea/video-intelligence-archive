# archive/web/app.py
import asyncio
import hashlib
from pathlib import Path
from fastapi import FastAPI, UploadFile, File
from fastapi.responses import JSONResponse, HTMLResponse
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
    return get_store().list_videos()

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
