# 멀티모달 영상 아카이브 & 설명 가능한 검색 — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 영상을 업로드하면 Whisper STT + GPT Vision으로 색인하고, 키워드(FTS5)와 의미(임베딩)를 합산한 하이브리드 검색에 출처·이유 설명을 붙여 반환하는 영속 아카이브 웹서비스를 만든다.

**Architecture:** FastAPI 웹 + 백그라운드 처리 파이프라인. 파이프라인은 단계별 체크포인트로 실패해도 부분 색인하고 재시작 시 재개. 저장은 SQLite(FTS5) + ChromaDB 이중. 검색 엔진은 웹과 텔레그램 봇이 공유. 모든 외부 클라이언트(OpenAI/Chroma/Telegram)는 사용 시점 지연 초기화.

**Tech Stack:** Python 3.12, FastAPI, uvicorn, openai, chromadb, python-dotenv, httpx, sse-starlette, pytest, 시스템 ffmpeg.

---

## File Structure

```
archive/
├── __init__.py
├── config.py            # .env 로드 + 지연 초기화 팩토리
├── events.py            # 인프로세스 진행 이벤트 버스
├── media.py             # ffmpeg: 오디오 추출, keyframe 추출
├── transcribe.py        # Whisper STT → 세그먼트
├── embed.py             # text-embedding-3-small
├── enrich/
│   ├── __init__.py
│   ├── subtitle.py      # GPT 자막 교정
│   └── vision.py        # GPT Vision 장면분석 + 금칙
├── store/
│   ├── __init__.py
│   ├── sqlite_store.py  # 테이블 + FTS5
│   └── vector_store.py  # ChromaDB
├── pipeline.py          # 단계 오케스트레이션
├── runbook.py           # 운영 절차서 .md
├── search/
│   ├── __init__.py
│   ├── query.py         # GPT 쿼리 분해
│   ├── engine.py        # 4모드 하이브리드
│   └── explain.py       # 결정론적 설명
├── notify/
│   ├── __init__.py
│   └── telegram.py      # 알림 + 봇 검색
└── web/
    ├── __init__.py
    ├── app.py           # FastAPI 라우트
    └── static/index.html
tests/                   # 모듈별 단위 + 통합 테스트
```

각 파일은 단일 책임. 하위 모듈은 상위를 모름 → 외부 의존을 목으로 대체 가능.

---

## Task 0: 프로젝트 스캐폴드

**Files:**
- Create: `requirements.txt`, `pyproject.toml`, `.env.example`, `archive/__init__.py`, `tests/__init__.py`, `tests/conftest.py`

- [ ] **Step 1: requirements.txt 작성**

```
fastapi==0.115.*
uvicorn==0.32.*
openai==1.*
chromadb==0.5.*
python-dotenv==1.*
httpx==0.27.*
sse-starlette==2.*
pytest==8.*
```

- [ ] **Step 2: pyproject.toml (pytest 경로 설정)**

```toml
[tool.pytest.ini_options]
pythonpath = ["."]
testpaths = ["tests"]
```

- [ ] **Step 3: .env.example 작성**

```
OPENAI_API_KEY=
TELEGRAM_BOT_TOKEN=
TELEGRAM_CHAT_ID=
DATA_DIR=./data
HYBRID_ALPHA=0.5
KEYFRAME_THRESHOLD=0.3
```

- [ ] **Step 4: 빈 패키지 파일 생성**

`archive/__init__.py`, `tests/__init__.py` 는 빈 파일.

`tests/conftest.py`:
```python
import os
import pytest

@pytest.fixture
def tmp_data_dir(tmp_path, monkeypatch):
    monkeypatch.setenv("DATA_DIR", str(tmp_path))
    return tmp_path
```

- [ ] **Step 5: 가상환경 + 설치 + 커밋**

```bash
python3 -m venv .venv && . .venv/bin/activate && pip install -r requirements.txt
git add requirements.txt pyproject.toml .env.example archive/__init__.py tests/__init__.py tests/conftest.py
git commit -m "chore: project scaffold and dependencies"
```

---

## Task 1: config — 지연 초기화 (제약②④)

**Files:**
- Create: `archive/config.py`
- Test: `tests/test_config.py`

- [ ] **Step 1: 실패 테스트 작성**

```python
# tests/test_config.py
import pytest
import archive.config as config

def test_data_dir_from_env(tmp_data_dir):
    assert config.data_dir() == tmp_data_dir

def test_get_openai_raises_without_key(monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    config.reset()
    with pytest.raises(RuntimeError, match="OPENAI_API_KEY"):
        config.get_openai()

def test_get_openai_lazy_caches(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    config.reset()
    calls = []
    monkeypatch.setattr(config, "_make_openai", lambda: calls.append(1) or object())
    a = config.get_openai()
    b = config.get_openai()
    assert a is b and len(calls) == 1

def test_hybrid_alpha_default(monkeypatch):
    monkeypatch.delenv("HYBRID_ALPHA", raising=False)
    assert config.hybrid_alpha() == 0.5
```

- [ ] **Step 2: 실패 확인**

Run: `pytest tests/test_config.py -v`
Expected: FAIL (module attrs not defined)

- [ ] **Step 3: 구현**

```python
# archive/config.py
import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

_openai = None
_chroma = None

def reset():
    """테스트용: 캐시 초기화."""
    global _openai, _chroma
    _openai = None
    _chroma = None

def data_dir() -> Path:
    p = Path(os.environ.get("DATA_DIR", "./data"))
    p.mkdir(parents=True, exist_ok=True)
    return p

def hybrid_alpha() -> float:
    return float(os.environ.get("HYBRID_ALPHA", "0.5"))

def keyframe_threshold() -> float:
    return float(os.environ.get("KEYFRAME_THRESHOLD", "0.3"))

def _make_openai():
    from openai import OpenAI
    return OpenAI()  # OPENAI_API_KEY 환경변수 사용

def get_openai():
    global _openai
    if _openai is None:
        if not os.environ.get("OPENAI_API_KEY"):
            raise RuntimeError("OPENAI_API_KEY is not set")
        _openai = _make_openai()
    return _openai

def get_chroma():
    global _chroma
    if _chroma is None:
        import chromadb
        _chroma = chromadb.PersistentClient(path=str(data_dir() / "chroma"))
    return _chroma

def telegram_token() -> str | None:
    return os.environ.get("TELEGRAM_BOT_TOKEN") or None

def telegram_chat_id() -> str | None:
    return os.environ.get("TELEGRAM_CHAT_ID") or None
```

- [ ] **Step 4: 통과 확인**

Run: `pytest tests/test_config.py -v`
Expected: PASS

- [ ] **Step 5: 커밋**

```bash
git add archive/config.py tests/test_config.py
git commit -m "feat: config with lazy client initialization"
```

---

## Task 2: events — 진행 이벤트 버스

**Files:**
- Create: `archive/events.py`
- Test: `tests/test_events.py`

- [ ] **Step 1: 실패 테스트 작성**

```python
# tests/test_events.py
from archive.events import EventBus, Event

def test_subscribe_receives_published_event():
    bus = EventBus()
    received = []
    bus.subscribe(received.append)
    ev = Event(video_id=1, step="transcribe", status="done", message="ok")
    bus.publish(ev)
    assert received == [ev]

def test_multiple_subscribers_all_notified():
    bus = EventBus()
    a, b = [], []
    bus.subscribe(a.append)
    bus.subscribe(b.append)
    bus.publish(Event(video_id=1, step="audio", status="running", message=""))
    assert len(a) == 1 and len(b) == 1

def test_subscriber_error_does_not_break_others():
    bus = EventBus()
    good = []
    bus.subscribe(lambda e: (_ for _ in ()).throw(ValueError()))
    bus.subscribe(good.append)
    bus.publish(Event(video_id=1, step="x", status="failed", message="boom"))
    assert len(good) == 1
```

- [ ] **Step 2: 실패 확인**

Run: `pytest tests/test_events.py -v`
Expected: FAIL (no module)

- [ ] **Step 3: 구현**

```python
# archive/events.py
from dataclasses import dataclass
from typing import Callable

@dataclass(frozen=True)
class Event:
    video_id: int
    step: str
    status: str        # running|done|failed|skipped|info
    message: str

class EventBus:
    def __init__(self):
        self._subs: list[Callable[[Event], None]] = []

    def subscribe(self, fn: Callable[[Event], None]) -> None:
        self._subs.append(fn)

    def publish(self, event: Event) -> None:
        for fn in list(self._subs):
            try:
                fn(event)
            except Exception:
                pass  # 한 구독자 실패가 다른 구독자를 막지 않음

bus = EventBus()  # 전역 기본 버스
```

- [ ] **Step 4: 통과 확인**

Run: `pytest tests/test_events.py -v`
Expected: PASS

- [ ] **Step 5: 커밋**

```bash
git add archive/events.py tests/test_events.py
git commit -m "feat: in-process event bus for progress fan-out"
```

---

## Task 3: store/sqlite_store — 테이블 + FTS5 (제약①)

**Files:**
- Create: `archive/store/__init__.py` (빈 파일), `archive/store/sqlite_store.py`
- Test: `tests/test_sqlite_store.py`

- [ ] **Step 1: 실패 테스트 작성**

```python
# tests/test_sqlite_store.py
from archive.store.sqlite_store import SqliteStore

def test_insert_and_get_video(tmp_data_dir):
    s = SqliteStore(tmp_data_dir / "t.db")
    vid = s.insert_video(filename="a.mp4", stored_path="/x/a.mp4", sha256="abc")
    v = s.get_video(vid)
    assert v["filename"] == "a.mp4" and v["status"] == "pending"

def test_dedup_returns_existing_video(tmp_data_dir):
    s = SqliteStore(tmp_data_dir / "t.db")
    a = s.insert_video("a.mp4", "/x/a.mp4", "samehash")
    b = s.insert_video("a.mp4", "/x/a.mp4", "samehash")
    assert a == b

def test_insert_chunk_and_fts_search(tmp_data_dir):
    s = SqliteStore(tmp_data_dir / "t.db")
    vid = s.insert_video("a.mp4", "/x/a.mp4", "h1")
    s.insert_chunk(vid, kind="transcript", seq=0, start_s=0.0, end_s=2.0,
                   text="강아지가 공원에서 뛰어논다")
    hits = s.fts_search("강아지", limit=5)
    assert len(hits) == 1 and hits[0]["text"].startswith("강아지")
    assert "bm25" in hits[0]

def test_persistence_across_reopen(tmp_data_dir):
    path = tmp_data_dir / "t.db"
    s1 = SqliteStore(path)
    vid = s1.insert_video("a.mp4", "/x/a.mp4", "h1")
    s1.insert_chunk(vid, "transcript", 0, 0.0, 1.0, "hello world")
    s1.close()
    s2 = SqliteStore(path)
    assert len(s2.fts_search("hello")) == 1

def test_job_checkpoint_roundtrip(tmp_data_dir):
    s = SqliteStore(tmp_data_dir / "t.db")
    vid = s.insert_video("a.mp4", "/x/a.mp4", "h1")
    s.set_job(vid, "transcribe", "done")
    s.set_job(vid, "vision", "failed", error="api error")
    assert s.last_completed_step(vid) == "transcribe"
    assert s.get_jobs(vid)["vision"]["status"] == "failed"

def test_flag_attached_to_chunk(tmp_data_dir):
    s = SqliteStore(tmp_data_dir / "t.db")
    vid = s.insert_video("a.mp4", "/x/a.mp4", "h1")
    cid = s.insert_chunk(vid, "scene", 0, 0.0, 1.0, "폭력적 장면", frame_path="/f/0.jpg")
    s.add_flag(cid, category="violence", severity="high", note="weapon")
    flags = s.flags_for_chunk(cid)
    assert flags[0]["category"] == "violence"
```

- [ ] **Step 2: 실패 확인**

Run: `pytest tests/test_sqlite_store.py -v`
Expected: FAIL (no module)

- [ ] **Step 3: 구현**

```python
# archive/store/sqlite_store.py
import sqlite3
from pathlib import Path

STEP_ORDER = ["persist", "audio", "transcribe", "subtitle", "keyframes",
              "vision", "embed", "index-sql", "index-vec", "finalize"]

SCHEMA = """
CREATE TABLE IF NOT EXISTS videos(
  id INTEGER PRIMARY KEY, filename TEXT, stored_path TEXT, sha256 TEXT UNIQUE,
  duration_s REAL, status TEXT DEFAULT 'pending', error TEXT,
  created_at TEXT DEFAULT (datetime('now')), updated_at TEXT DEFAULT (datetime('now')));

CREATE TABLE IF NOT EXISTS chunks(
  id INTEGER PRIMARY KEY, video_id INTEGER, kind TEXT, seq INTEGER,
  start_s REAL, end_s REAL, text TEXT, frame_path TEXT,
  embedded INTEGER DEFAULT 0, created_at TEXT DEFAULT (datetime('now')));

CREATE TABLE IF NOT EXISTS chunk_flags(
  id INTEGER PRIMARY KEY, chunk_id INTEGER, category TEXT, severity TEXT, note TEXT);

CREATE TABLE IF NOT EXISTS jobs(
  id INTEGER PRIMARY KEY, video_id INTEGER, step TEXT, status TEXT,
  error TEXT, updated_at TEXT DEFAULT (datetime('now')),
  UNIQUE(video_id, step));

CREATE VIRTUAL TABLE IF NOT EXISTS chunks_fts USING fts5(
  text, content='chunks', content_rowid='id');

CREATE TRIGGER IF NOT EXISTS chunks_ai AFTER INSERT ON chunks BEGIN
  INSERT INTO chunks_fts(rowid, text) VALUES (new.id, new.text);
END;
CREATE TRIGGER IF NOT EXISTS chunks_ad AFTER DELETE ON chunks BEGIN
  INSERT INTO chunks_fts(chunks_fts, rowid, text) VALUES('delete', old.id, old.text);
END;
"""

class SqliteStore:
    def __init__(self, path: Path):
        self.conn = sqlite3.connect(str(path), check_same_thread=False)
        self.conn.row_factory = sqlite3.Row
        self.conn.executescript(SCHEMA)
        self.conn.commit()

    def close(self):
        self.conn.close()

    # --- videos ---
    def insert_video(self, filename, stored_path, sha256) -> int:
        cur = self.conn.execute("SELECT id FROM videos WHERE sha256=?", (sha256,))
        row = cur.fetchone()
        if row:
            return row["id"]
        cur = self.conn.execute(
            "INSERT INTO videos(filename, stored_path, sha256) VALUES(?,?,?)",
            (filename, stored_path, sha256))
        self.conn.commit()
        return cur.lastrowid

    def get_video(self, vid) -> dict | None:
        row = self.conn.execute("SELECT * FROM videos WHERE id=?", (vid,)).fetchone()
        return dict(row) if row else None

    def set_video_status(self, vid, status, error=None):
        self.conn.execute(
            "UPDATE videos SET status=?, error=?, updated_at=datetime('now') WHERE id=?",
            (status, error, vid))
        self.conn.commit()

    def list_videos(self) -> list[dict]:
        rows = self.conn.execute("SELECT * FROM videos ORDER BY id DESC").fetchall()
        return [dict(r) for r in rows]

    def incomplete_videos(self) -> list[int]:
        rows = self.conn.execute(
            "SELECT id FROM videos WHERE status NOT IN ('done','failed')").fetchall()
        return [r["id"] for r in rows]

    # --- chunks ---
    def insert_chunk(self, video_id, kind, seq, start_s, end_s, text,
                     frame_path=None) -> int:
        cur = self.conn.execute(
            "INSERT INTO chunks(video_id,kind,seq,start_s,end_s,text,frame_path) "
            "VALUES(?,?,?,?,?,?,?)",
            (video_id, kind, seq, start_s, end_s, text, frame_path))
        self.conn.commit()
        return cur.lastrowid

    def mark_embedded(self, chunk_id):
        self.conn.execute("UPDATE chunks SET embedded=1 WHERE id=?", (chunk_id,))
        self.conn.commit()

    def chunks_for_video(self, video_id) -> list[dict]:
        rows = self.conn.execute(
            "SELECT * FROM chunks WHERE video_id=? ORDER BY seq", (video_id,)).fetchall()
        return [dict(r) for r in rows]

    def get_chunk(self, chunk_id) -> dict | None:
        row = self.conn.execute("SELECT * FROM chunks WHERE id=?", (chunk_id,)).fetchone()
        return dict(row) if row else None

    # --- fts ---
    def fts_search(self, query, limit=20) -> list[dict]:
        rows = self.conn.execute(
            "SELECT c.*, bm25(chunks_fts) AS bm25 "
            "FROM chunks_fts JOIN chunks c ON c.id = chunks_fts.rowid "
            "WHERE chunks_fts MATCH ? ORDER BY bm25 LIMIT ?",
            (query, limit)).fetchall()
        return [dict(r) for r in rows]

    # --- flags ---
    def add_flag(self, chunk_id, category, severity, note=None):
        self.conn.execute(
            "INSERT INTO chunk_flags(chunk_id,category,severity,note) VALUES(?,?,?,?)",
            (chunk_id, category, severity, note))
        self.conn.commit()

    def flags_for_chunk(self, chunk_id) -> list[dict]:
        rows = self.conn.execute(
            "SELECT * FROM chunk_flags WHERE chunk_id=?", (chunk_id,)).fetchall()
        return [dict(r) for r in rows]

    # --- jobs / checkpoint ---
    def set_job(self, video_id, step, status, error=None):
        self.conn.execute(
            "INSERT INTO jobs(video_id,step,status,error,updated_at) "
            "VALUES(?,?,?,?,datetime('now')) "
            "ON CONFLICT(video_id,step) DO UPDATE SET "
            "status=excluded.status, error=excluded.error, updated_at=datetime('now')",
            (video_id, step, status, error))
        self.conn.commit()

    def get_jobs(self, video_id) -> dict:
        rows = self.conn.execute(
            "SELECT step,status,error FROM jobs WHERE video_id=?", (video_id,)).fetchall()
        return {r["step"]: {"status": r["status"], "error": r["error"]} for r in rows}

    def last_completed_step(self, video_id) -> str | None:
        jobs = self.get_jobs(video_id)
        done = [s for s in STEP_ORDER if jobs.get(s, {}).get("status") == "done"]
        return done[-1] if done else None
```

- [ ] **Step 4: 통과 확인**

Run: `pytest tests/test_sqlite_store.py -v`
Expected: PASS

- [ ] **Step 5: 커밋**

```bash
git add archive/store/__init__.py archive/store/sqlite_store.py tests/test_sqlite_store.py
git commit -m "feat: SQLite store with FTS5, chunks, flags, job checkpoints"
```

---

## Task 4: store/vector_store — ChromaDB (제약①)

**Files:**
- Create: `archive/store/vector_store.py`
- Test: `tests/test_vector_store.py`

- [ ] **Step 1: 실패 테스트 작성**

```python
# tests/test_vector_store.py
import chromadb
from archive.store.vector_store import VectorStore

def make_store(tmp_path):
    client = chromadb.PersistentClient(path=str(tmp_path / "chroma"))
    return VectorStore(client)

def test_add_and_query(tmp_path):
    vs = make_store(tmp_path)
    vs.add(chunk_id=1, embedding=[1.0, 0.0], text="dog",
           metadata={"video_id": 1, "kind": "transcript", "start_s": 0.0,
                     "end_s": 1.0, "has_flags": False})
    vs.add(chunk_id=2, embedding=[0.0, 1.0], text="cat",
           metadata={"video_id": 1, "kind": "transcript", "start_s": 1.0,
                     "end_s": 2.0, "has_flags": False})
    res = vs.query([1.0, 0.0], k=2)
    assert res[0]["chunk_id"] == 1
    assert "distance" in res[0]

def test_persistence(tmp_path):
    vs1 = make_store(tmp_path)
    vs1.add(1, [1.0, 0.0], "dog", {"video_id": 1, "kind": "transcript",
            "start_s": 0.0, "end_s": 1.0, "has_flags": False})
    vs2 = make_store(tmp_path)
    assert vs2.query([1.0, 0.0], k=1)[0]["chunk_id"] == 1
```

- [ ] **Step 2: 실패 확인**

Run: `pytest tests/test_vector_store.py -v`
Expected: FAIL (no module)

- [ ] **Step 3: 구현**

```python
# archive/store/vector_store.py
class VectorStore:
    def __init__(self, client, collection_name="chunks"):
        self.col = client.get_or_create_collection(name=collection_name)

    def add(self, chunk_id, embedding, text, metadata):
        self.col.upsert(
            ids=[str(chunk_id)], embeddings=[embedding],
            documents=[text], metadatas=[metadata])

    def query(self, embedding, k=20, where=None):
        res = self.col.query(query_embeddings=[embedding], n_results=k, where=where)
        out = []
        ids = res["ids"][0]
        dists = res["distances"][0]
        docs = res["documents"][0]
        metas = res["metadatas"][0]
        for i in range(len(ids)):
            out.append({
                "chunk_id": int(ids[i]), "distance": dists[i],
                "text": docs[i], "metadata": metas[i],
            })
        return out
```

- [ ] **Step 4: 통과 확인**

Run: `pytest tests/test_vector_store.py -v`
Expected: PASS

- [ ] **Step 5: 커밋**

```bash
git add archive/store/vector_store.py tests/test_vector_store.py
git commit -m "feat: ChromaDB vector store with persistent upsert/query"
```

---

## Task 5: media — ffmpeg 오디오/keyframe 추출

**Files:**
- Create: `archive/media.py`
- Test: `tests/test_media.py`

ffmpeg는 subprocess로 호출. 테스트는 subprocess 명령 조립을 검증(실제 ffmpeg 미실행).

- [ ] **Step 1: 실패 테스트 작성**

```python
# tests/test_media.py
import archive.media as media

def test_extract_audio_builds_ffmpeg_cmd(monkeypatch, tmp_path):
    calls = {}
    def fake_run(cmd, **kw):
        calls["cmd"] = cmd
        (tmp_path / "out.wav").write_bytes(b"x")
        class R: returncode = 0
        return R()
    monkeypatch.setattr(media.subprocess, "run", fake_run)
    out = media.extract_audio(tmp_path / "in.mp4", tmp_path / "out.wav")
    assert out == tmp_path / "out.wav"
    assert "ffmpeg" in calls["cmd"][0]
    assert str(tmp_path / "in.mp4") in calls["cmd"]

def test_extract_keyframes_parses_frame_files(monkeypatch, tmp_path):
    frames_dir = tmp_path / "frames"
    def fake_run(cmd, **kw):
        frames_dir.mkdir(parents=True, exist_ok=True)
        (frames_dir / "0001.jpg").write_bytes(b"x")
        (frames_dir / "0002.jpg").write_bytes(b"x")
        class R: returncode = 0
        return R()
    monkeypatch.setattr(media.subprocess, "run", fake_run)
    frames = media.extract_keyframes(tmp_path / "in.mp4", frames_dir, threshold=0.3)
    assert len(frames) == 2
    assert all(p.suffix == ".jpg" for p in frames)
```

- [ ] **Step 2: 실패 확인**

Run: `pytest tests/test_media.py -v`
Expected: FAIL (no module)

- [ ] **Step 3: 구현**

```python
# archive/media.py
import subprocess
from pathlib import Path

def extract_audio(video_path: Path, out_path: Path) -> Path:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    cmd = ["ffmpeg", "-y", "-i", str(video_path), "-vn",
           "-ac", "1", "-ar", "16000", str(out_path)]
    subprocess.run(cmd, check=True, capture_output=True)
    return out_path

def extract_keyframes(video_path: Path, out_dir: Path, threshold: float = 0.3) -> list[Path]:
    """장면 전환 감지로 keyframe 추출. out_dir/NNNN.jpg 로 저장."""
    out_dir.mkdir(parents=True, exist_ok=True)
    cmd = ["ffmpeg", "-y", "-i", str(video_path),
           "-vf", f"select='gt(scene,{threshold})',showinfo",
           "-vsync", "vfr", str(out_dir / "%04d.jpg")]
    subprocess.run(cmd, check=True, capture_output=True)
    return sorted(out_dir.glob("*.jpg"))
```

- [ ] **Step 4: 통과 확인**

Run: `pytest tests/test_media.py -v`
Expected: PASS

- [ ] **Step 5: 커밋**

```bash
git add archive/media.py tests/test_media.py
git commit -m "feat: ffmpeg audio extraction and scene-change keyframes"
```

---

## Task 6: transcribe — Whisper STT

**Files:**
- Create: `archive/transcribe.py`
- Test: `tests/test_transcribe.py`

- [ ] **Step 1: 실패 테스트 작성**

```python
# tests/test_transcribe.py
import archive.transcribe as transcribe

class FakeSeg:
    def __init__(self, start, end, text):
        self.start, self.end, self.text = start, end, text

class FakeResult:
    segments = [FakeSeg(0.0, 2.0, "안녕하세요"), FakeSeg(2.0, 4.0, "반갑습니다")]

def test_transcribe_returns_segments(monkeypatch, tmp_path):
    audio = tmp_path / "a.wav"; audio.write_bytes(b"x")
    fake_client = type("C", (), {})()
    fake_client.audio = type("A", (), {})()
    fake_client.audio.transcriptions = type("T", (), {
        "create": staticmethod(lambda **kw: FakeResult())})()
    monkeypatch.setattr(transcribe.config, "get_openai", lambda: fake_client)
    segs = transcribe.transcribe(audio)
    assert segs == [
        {"start": 0.0, "end": 2.0, "text": "안녕하세요"},
        {"start": 2.0, "end": 4.0, "text": "반갑습니다"},
    ]
```

- [ ] **Step 2: 실패 확인**

Run: `pytest tests/test_transcribe.py -v`
Expected: FAIL

- [ ] **Step 3: 구현**

```python
# archive/transcribe.py
from pathlib import Path
from archive import config

def transcribe(audio_path: Path) -> list[dict]:
    client = config.get_openai()
    with open(audio_path, "rb") as f:
        result = client.audio.transcriptions.create(
            model="whisper-1", file=f, response_format="verbose_json",
            timestamp_granularities=["segment"])
    return [{"start": float(s.start), "end": float(s.end), "text": s.text.strip()}
            for s in result.segments]
```

- [ ] **Step 4: 통과 확인**

Run: `pytest tests/test_transcribe.py -v`
Expected: PASS

- [ ] **Step 5: 커밋**

```bash
git add archive/transcribe.py tests/test_transcribe.py
git commit -m "feat: Whisper STT to timestamped segments"
```

---

## Task 7: embed — text-embedding-3-small

**Files:**
- Create: `archive/embed.py`
- Test: `tests/test_embed.py`

- [ ] **Step 1: 실패 테스트 작성**

```python
# tests/test_embed.py
import archive.embed as embed

def test_embed_texts_returns_vectors(monkeypatch):
    class Item:
        def __init__(self, v): self.embedding = v
    class Resp:
        data = [Item([0.1, 0.2]), Item([0.3, 0.4])]
    fake = type("C", (), {})()
    fake.embeddings = type("E", (), {
        "create": staticmethod(lambda **kw: Resp())})()
    monkeypatch.setattr(embed.config, "get_openai", lambda: fake)
    out = embed.embed_texts(["a", "b"])
    assert out == [[0.1, 0.2], [0.3, 0.4]]

def test_embed_empty_returns_empty(monkeypatch):
    out = embed.embed_texts([])
    assert out == []
```

- [ ] **Step 2: 실패 확인**

Run: `pytest tests/test_embed.py -v`
Expected: FAIL

- [ ] **Step 3: 구현**

```python
# archive/embed.py
from archive import config

MODEL = "text-embedding-3-small"

def embed_texts(texts: list[str]) -> list[list[float]]:
    if not texts:
        return []
    client = config.get_openai()
    resp = client.embeddings.create(model=MODEL, input=texts)
    return [item.embedding for item in resp.data]

def embed_one(text: str) -> list[float]:
    return embed_texts([text])[0]
```

- [ ] **Step 4: 통과 확인**

Run: `pytest tests/test_embed.py -v`
Expected: PASS

- [ ] **Step 5: 커밋**

```bash
git add archive/embed.py tests/test_embed.py
git commit -m "feat: text-embedding-3-small batch embeddings"
```

---

## Task 8: enrich/subtitle — GPT 자막 교정

**Files:**
- Create: `archive/enrich/__init__.py` (빈 파일), `archive/enrich/subtitle.py`
- Test: `tests/test_subtitle.py`

- [ ] **Step 1: 실패 테스트 작성**

```python
# tests/test_subtitle.py
import json
import archive.enrich.subtitle as subtitle

def fake_client(content):
    c = type("C", (), {})()
    msg = type("M", (), {"content": content})()
    choice = type("Ch", (), {"message": msg})()
    resp = type("R", (), {"choices": [choice]})()
    c.chat = type("Chat", (), {})()
    c.chat.completions = type("CC", (), {
        "create": staticmethod(lambda **kw: resp)})()
    return c

def test_correct_returns_corrected_texts(monkeypatch):
    payload = json.dumps({"texts": ["안녕하세요.", "반갑습니다."]})
    monkeypatch.setattr(subtitle.config, "get_openai", lambda: fake_client(payload))
    segs = [{"start": 0, "end": 1, "text": "안뇽하세요"},
            {"start": 1, "end": 2, "text": "반갑슴니다"}]
    out = subtitle.correct(segs)
    assert [s["text"] for s in out] == ["안녕하세요.", "반갑습니다."]
    assert out[0]["start"] == 0  # 타임스탬프 보존

def test_correct_empty_returns_empty(monkeypatch):
    assert subtitle.correct([]) == []
```

- [ ] **Step 2: 실패 확인**

Run: `pytest tests/test_subtitle.py -v`
Expected: FAIL

- [ ] **Step 3: 구현**

```python
# archive/enrich/subtitle.py
import json
from archive import config

MODEL = "gpt-5.5"

SYSTEM = ("너는 자막 교정기다. 입력 자막 텍스트 배열의 맞춤법/띄어쓰기/오탈자만 교정하고 "
          "의미는 바꾸지 마라. 원소 개수와 순서를 유지해 "
          '{"texts": [...]} JSON으로만 답하라.')

def correct(segments: list[dict]) -> list[dict]:
    if not segments:
        return []
    client = config.get_openai()
    texts = [s["text"] for s in segments]
    resp = client.chat.completions.create(
        model=MODEL,
        response_format={"type": "json_object"},
        messages=[{"role": "system", "content": SYSTEM},
                  {"role": "user", "content": json.dumps({"texts": texts},
                                                          ensure_ascii=False)}])
    corrected = json.loads(resp.choices[0].message.content)["texts"]
    if len(corrected) != len(segments):
        return segments  # 개수 불일치 시 원본 유지 (안전)
    return [{**s, "text": corrected[i]} for i, s in enumerate(segments)]
```

- [ ] **Step 4: 통과 확인**

Run: `pytest tests/test_subtitle.py -v`
Expected: PASS

- [ ] **Step 5: 커밋**

```bash
git add archive/enrich/__init__.py archive/enrich/subtitle.py tests/test_subtitle.py
git commit -m "feat: GPT subtitle correction preserving timestamps"
```

---

## Task 9: enrich/vision — GPT Vision 장면분석 + 금칙

**Files:**
- Create: `archive/enrich/vision.py`
- Test: `tests/test_vision.py`

- [ ] **Step 1: 실패 테스트 작성**

```python
# tests/test_vision.py
import json
import archive.enrich.vision as vision

def fake_client(content):
    c = type("C", (), {})()
    msg = type("M", (), {"content": content})()
    choice = type("Ch", (), {"message": msg})()
    resp = type("R", (), {"choices": [choice]})()
    c.chat = type("Chat", (), {})()
    c.chat.completions = type("CC", (), {
        "create": staticmethod(lambda **kw: resp)})()
    return c

def test_analyze_frame_returns_description_and_flags(monkeypatch, tmp_path):
    frame = tmp_path / "0001.jpg"; frame.write_bytes(b"\xff\xd8\xff")
    payload = json.dumps({
        "description": "공원에서 강아지가 뛰어논다",
        "flags": [{"category": "none", "severity": "low", "note": ""}]})
    monkeypatch.setattr(vision.config, "get_openai", lambda: fake_client(payload))
    res = vision.analyze_frame(frame)
    assert res["description"].startswith("공원")
    assert res["flags"][0]["category"] == "none"

def test_analyze_frame_detects_prohibited(monkeypatch, tmp_path):
    frame = tmp_path / "0002.jpg"; frame.write_bytes(b"\xff\xd8\xff")
    payload = json.dumps({
        "description": "폭력적 장면",
        "flags": [{"category": "violence", "severity": "high", "note": "무기"}]})
    monkeypatch.setattr(vision.config, "get_openai", lambda: fake_client(payload))
    res = vision.analyze_frame(frame)
    assert res["flags"][0]["category"] == "violence"
```

- [ ] **Step 2: 실패 확인**

Run: `pytest tests/test_vision.py -v`
Expected: FAIL

- [ ] **Step 3: 구현**

```python
# archive/enrich/vision.py
import base64
import json
from pathlib import Path
from archive import config

MODEL = "gpt-5.5"

SYSTEM = ("너는 영상 프레임 분석기다. 이미지를 보고 (1) 장면을 한국어로 1~2문장 묘사하고 "
          "(2) 금칙 요소(선정성 sexual, 폭력 violence, 혐오 hate, 로고/저작권 logo)를 "
          'category/severity(low|medium|high)/note 로 태깅하라. 해당 없으면 category="none". '
          '반드시 {"description": str, "flags": [{"category","severity","note"}]} JSON으로만 답하라.')

def analyze_frame(frame_path: Path) -> dict:
    client = config.get_openai()
    b64 = base64.b64encode(Path(frame_path).read_bytes()).decode()
    resp = client.chat.completions.create(
        model=MODEL,
        response_format={"type": "json_object"},
        messages=[
            {"role": "system", "content": SYSTEM},
            {"role": "user", "content": [
                {"type": "text", "text": "이 프레임을 분석하라."},
                {"type": "image_url",
                 "image_url": {"url": f"data:image/jpeg;base64,{b64}"}}]}])
    data = json.loads(resp.choices[0].message.content)
    data.setdefault("flags", [])
    return data

def real_flags(flags: list[dict]) -> list[dict]:
    """category='none' 제외한 실제 금칙만."""
    return [f for f in flags if f.get("category") and f["category"] != "none"]
```

- [ ] **Step 4: 통과 확인**

Run: `pytest tests/test_vision.py -v`
Expected: PASS

- [ ] **Step 5: 커밋**

```bash
git add archive/enrich/vision.py tests/test_vision.py
git commit -m "feat: GPT Vision scene analysis with prohibited-content tagging"
```

---

## Task 10: search/explain — 결정론적 설명

**Files:**
- Create: `archive/search/__init__.py` (빈 파일), `archive/search/explain.py`
- Test: `tests/test_explain.py`

- [ ] **Step 1: 실패 테스트 작성**

```python
# tests/test_explain.py
from archive.search.explain import build_explanation

def test_both_source():
    e = build_explanation(matched_terms=["강아지"], bm25=3.2, similarity=0.81)
    assert e["source"] == "both"
    assert "강아지" in e["text"]
    assert "0.81" in e["text"]

def test_keyword_only():
    e = build_explanation(matched_terms=["공원"], bm25=2.0, similarity=None)
    assert e["source"] == "keyword"
    assert "공원" in e["text"]

def test_vector_only():
    e = build_explanation(matched_terms=[], bm25=None, similarity=0.7)
    assert e["source"] == "vector"
    assert "0.70" in e["text"]
```

- [ ] **Step 2: 실패 확인**

Run: `pytest tests/test_explain.py -v`
Expected: FAIL

- [ ] **Step 3: 구현**

```python
# archive/search/explain.py
def build_explanation(matched_terms, bm25, similarity) -> dict:
    has_kw = bm25 is not None
    has_vec = similarity is not None
    if has_kw and has_vec:
        source = "both"
    elif has_kw:
        source = "keyword"
    else:
        source = "vector"

    parts = []
    if has_kw:
        terms = ", ".join(f"'{t}'" for t in matched_terms) if matched_terms else "키워드"
        parts.append(f"{terms} 일치 (FTS {bm25:.1f})")
    if has_vec:
        parts.append(f"의미 유사도 {similarity:.2f}")
    text = " + ".join(parts) + f" → {source}"
    return {"source": source, "matched_terms": matched_terms,
            "bm25": bm25, "similarity": similarity, "text": text}
```

- [ ] **Step 4: 통과 확인**

Run: `pytest tests/test_explain.py -v`
Expected: PASS

- [ ] **Step 5: 커밋**

```bash
git add archive/search/__init__.py archive/search/explain.py tests/test_explain.py
git commit -m "feat: deterministic search-result explanations"
```

---

## Task 11: search/engine — 4모드 하이브리드

**Files:**
- Create: `archive/search/engine.py`
- Test: `tests/test_engine.py`

`SearchEngine`는 `SqliteStore`·`VectorStore`·임베딩 함수를 주입받아 테스트 가능.

- [ ] **Step 1: 실패 테스트 작성**

```python
# tests/test_engine.py
from archive.store.sqlite_store import SqliteStore
from archive.search.engine import SearchEngine

class FakeVector:
    def __init__(self, results): self._r = results
    def query(self, embedding, k=20, where=None): return self._r

def seed(tmp_path):
    s = SqliteStore(tmp_path / "t.db")
    vid = s.insert_video("a.mp4", "/x/a.mp4", "h1")
    c1 = s.insert_chunk(vid, "transcript", 0, 0.0, 2.0, "강아지가 공원에서 뛴다")
    c2 = s.insert_chunk(vid, "transcript", 1, 2.0, 4.0, "고양이가 잔다")
    return s, c1, c2

def test_keyword_mode(tmp_path):
    s, c1, c2 = seed(tmp_path)
    eng = SearchEngine(s, FakeVector([]), embed_fn=lambda q: [0.0])
    res = eng.search("강아지", mode="keyword")
    assert res[0]["chunk_id"] == c1
    assert res[0]["explanation"]["source"] == "keyword"

def test_vector_mode(tmp_path):
    s, c1, c2 = seed(tmp_path)
    vec = FakeVector([{"chunk_id": c2, "distance": 0.1, "text": "고양이가 잔다",
                       "metadata": {}}])
    eng = SearchEngine(s, vec, embed_fn=lambda q: [0.0])
    res = eng.search("잠자는 동물", mode="vector")
    assert res[0]["chunk_id"] == c2
    assert res[0]["explanation"]["source"] == "vector"

def test_hybrid_marks_both_when_overlap(tmp_path):
    s, c1, c2 = seed(tmp_path)
    vec = FakeVector([{"chunk_id": c1, "distance": 0.2, "text": "강아지가 공원에서 뛴다",
                       "metadata": {}}])
    eng = SearchEngine(s, vec, embed_fn=lambda q: [0.0])
    res = eng.search("강아지", mode="hybrid")
    top = next(r for r in res if r["chunk_id"] == c1)
    assert top["explanation"]["source"] == "both"

def test_filter_mode_by_kind(tmp_path):
    s, c1, c2 = seed(tmp_path)
    s.insert_chunk(s.get_chunk(c1)["video_id"], "scene", 2, 4.0, 5.0, "장면")
    eng = SearchEngine(s, FakeVector([]), embed_fn=lambda q: [0.0])
    res = eng.search("", mode="filter", filters={"kind": "scene"})
    assert all(r["kind"] == "scene" for r in res)
```

- [ ] **Step 2: 실패 확인**

Run: `pytest tests/test_engine.py -v`
Expected: FAIL

- [ ] **Step 3: 구현**

```python
# archive/search/engine.py
from archive import config
from archive.search.explain import build_explanation

def _minmax(scores: dict) -> dict:
    if not scores:
        return {}
    vals = list(scores.values())
    lo, hi = min(vals), max(vals)
    if hi == lo:
        return {k: 1.0 for k in scores}
    return {k: (v - lo) / (hi - lo) for k, v in scores.items()}

class SearchEngine:
    def __init__(self, sqlite_store, vector_store, embed_fn):
        self.sql = sqlite_store
        self.vec = vector_store
        self.embed_fn = embed_fn

    def search(self, query, mode="hybrid", filters=None, limit=20):
        filters = filters or {}
        if mode == "filter":
            return self._filter(filters, limit)
        kw = self._keyword_scores(query, limit) if mode in ("keyword", "hybrid") else {}
        vec = self._vector_scores(query, limit, filters) if mode in ("vector", "hybrid") else {}
        return self._assemble(kw, vec, mode, limit)

    def _keyword_scores(self, query, limit):
        if not query:
            return {}
        hits = self.sql.fts_search(query, limit=limit)
        # bm25는 낮을수록 좋음 → 부호 반전해 점수화
        return {h["id"]: {"row": h, "bm25": h["bm25"], "raw": -h["bm25"]} for h in hits}

    def _vector_scores(self, query, limit, filters):
        if not query:
            return {}
        emb = self.embed_fn(query)
        where = {"kind": filters["kind"]} if filters.get("kind") else None
        hits = self.vec.query(emb, k=limit, where=where)
        out = {}
        for h in hits:
            sim = 1.0 / (1.0 + h["distance"])
            out[h["chunk_id"]] = {"sim": sim, "raw": sim}
        return out

    def _assemble(self, kw, vec, mode, limit):
        alpha = config.hybrid_alpha()
        kw_norm = _minmax({k: v["raw"] for k, v in kw.items()})
        vec_norm = _minmax({k: v["raw"] for k, v in vec.items()})
        ids = set(kw) | set(vec)
        results = []
        for cid in ids:
            kn = kw_norm.get(cid, 0.0)
            vn = vec_norm.get(cid, 0.0)
            if mode == "keyword":
                score = kn
            elif mode == "vector":
                score = vn
            else:
                score = alpha * vn + (1 - alpha) * kn
            row = self.sql.get_chunk(cid)
            if not row:
                continue
            matched = self._matched_terms(row["text"], kw.get(cid))
            expl = build_explanation(
                matched_terms=matched,
                bm25=kw[cid]["bm25"] if cid in kw else None,
                similarity=vec[cid]["sim"] if cid in vec else None)
            results.append(self._format(row, score, expl))
        results.sort(key=lambda r: r["score"], reverse=True)
        return results[:limit]

    def _matched_terms(self, text, kw_entry):
        if not kw_entry:
            return []
        return [w for w in set(text.split()) if w in text]  # 단순 텀 표기

    def _filter(self, filters, limit):
        clauses, params = [], []
        if filters.get("kind"):
            clauses.append("kind=?"); params.append(filters["kind"])
        if filters.get("video_id"):
            clauses.append("video_id=?"); params.append(filters["video_id"])
        if filters.get("flagged"):
            clauses.append("id IN (SELECT chunk_id FROM chunk_flags)")
        where = (" WHERE " + " AND ".join(clauses)) if clauses else ""
        rows = self.sql.conn.execute(
            f"SELECT * FROM chunks{where} ORDER BY video_id, seq LIMIT ?",
            (*params, limit)).fetchall()
        return [self._format(dict(r), score=None,
                expl={"source": "filter", "text": "필터 조건 일치"}) for r in rows]

    def _format(self, row, score, expl):
        flags = self.sql.flags_for_chunk(row["id"])
        return {
            "chunk_id": row["id"], "video_id": row["video_id"], "kind": row["kind"],
            "start_s": row["start_s"], "end_s": row["end_s"], "text": row["text"],
            "flags": flags, "score": score, "source": expl["source"],
            "explanation": expl,
        }
```

- [ ] **Step 4: 통과 확인**

Run: `pytest tests/test_engine.py -v`
Expected: PASS

- [ ] **Step 5: 커밋**

```bash
git add archive/search/engine.py tests/test_engine.py
git commit -m "feat: hybrid search engine with 4 modes and explanations"
```

---

## Task 12: search/query — GPT 쿼리 분해 (폴백 포함)

**Files:**
- Create: `archive/search/query.py`
- Test: `tests/test_query.py`

- [ ] **Step 1: 실패 테스트 작성**

```python
# tests/test_query.py
import json
import archive.search.query as query

def fake_client(content):
    c = type("C", (), {})()
    msg = type("M", (), {"content": content})()
    choice = type("Ch", (), {"message": msg})()
    resp = type("R", (), {"choices": [choice]})()
    c.chat = type("Chat", (), {})()
    c.chat.completions = type("CC", (), {
        "create": staticmethod(lambda **kw: resp)})()
    return c

def test_decompose_parses_structure(monkeypatch):
    payload = json.dumps({"keywords": ["강아지", "공원"],
                          "semantic_query": "공원에서 노는 강아지",
                          "filters": {"kind": "scene"}})
    monkeypatch.setattr(query.config, "get_openai", lambda: fake_client(payload))
    out = query.decompose("공원에서 강아지 노는 장면 찾아줘")
    assert out["keywords"] == ["강아지", "공원"]
    assert out["filters"]["kind"] == "scene"

def test_decompose_falls_back_on_error(monkeypatch):
    def boom(): raise RuntimeError("no key")
    monkeypatch.setattr(query.config, "get_openai", boom)
    out = query.decompose("강아지")
    assert out["keywords"] == ["강아지"]
    assert out["semantic_query"] == "강아지"
    assert out["filters"] == {}
```

- [ ] **Step 2: 실패 확인**

Run: `pytest tests/test_query.py -v`
Expected: FAIL

- [ ] **Step 3: 구현**

```python
# archive/search/query.py
import json
from archive import config

MODEL = "gpt-5.5"

SYSTEM = ("너는 검색 쿼리 분해기다. 사용자의 자연어 검색을 "
          '{"keywords": [핵심 키워드], "semantic_query": "의미 검색용 문장", '
          '"filters": {"kind": "transcript|scene", "flagged": true}} 형태 JSON으로 분해하라. '
          "해당 없는 필터는 생략. JSON으로만 답하라.")

def decompose(text: str) -> dict:
    try:
        client = config.get_openai()
        resp = client.chat.completions.create(
            model=MODEL, response_format={"type": "json_object"},
            messages=[{"role": "system", "content": SYSTEM},
                      {"role": "user", "content": text}])
        data = json.loads(resp.choices[0].message.content)
        return {"keywords": data.get("keywords", [text]),
                "semantic_query": data.get("semantic_query", text),
                "filters": data.get("filters", {})}
    except Exception:
        # 제약③ 정신: 분해 실패해도 원본 쿼리로 검색 가능
        return {"keywords": [text], "semantic_query": text, "filters": {}}
```

- [ ] **Step 4: 통과 확인**

Run: `pytest tests/test_query.py -v`
Expected: PASS

- [ ] **Step 5: 커밋**

```bash
git add archive/search/query.py tests/test_query.py
git commit -m "feat: GPT query decomposition with safe fallback"
```

---

## Task 13: pipeline — 단계 오케스트레이션 (제약①③)

**Files:**
- Create: `archive/pipeline.py`
- Test: `tests/test_pipeline.py`

파이프라인은 의존성(media/transcribe/subtitle/vision/embed/stores)을 모듈 함수로 호출.
테스트는 각 모듈 함수를 monkeypatch하고, **vision 실패 주입 시에도 transcript 청크가 색인됨**을 검증.

- [ ] **Step 1: 실패 테스트 작성**

```python
# tests/test_pipeline.py
import archive.pipeline as pipeline
from archive.store.sqlite_store import SqliteStore

class FakeVector:
    def __init__(self): self.added = []
    def add(self, chunk_id, embedding, text, metadata):
        self.added.append(chunk_id)

def patch_all(monkeypatch, tmp_path, vision_fails=False):
    monkeypatch.setattr(pipeline.media, "extract_audio",
                        lambda v, o: (o.write_bytes(b"x") or o))
    monkeypatch.setattr(pipeline.media, "extract_keyframes",
                        lambda v, d, threshold=0.3: [tmp_path / "f0.jpg"])
    (tmp_path / "f0.jpg").write_bytes(b"x")
    monkeypatch.setattr(pipeline.transcribe, "transcribe",
                        lambda a: [{"start": 0.0, "end": 2.0, "text": "강아지"}])
    monkeypatch.setattr(pipeline.subtitle, "correct", lambda segs: segs)
    def vision_fn(frame):
        if vision_fails:
            raise RuntimeError("vision down")
        return {"description": "공원 강아지",
                "flags": [{"category": "none", "severity": "low", "note": ""}]}
    monkeypatch.setattr(pipeline.vision, "analyze_frame", vision_fn)
    monkeypatch.setattr(pipeline.embed, "embed_texts",
                        lambda texts: [[0.1, 0.2] for _ in texts])

def test_full_pipeline_indexes_all(monkeypatch, tmp_path):
    patch_all(monkeypatch, tmp_path)
    s = SqliteStore(tmp_path / "t.db"); vec = FakeVector()
    vid = s.insert_video("a.mp4", str(tmp_path / "a.mp4"), "h1")
    (tmp_path / "a.mp4").write_bytes(b"x")
    pipeline.process_video(vid, s, vec, data_dir=tmp_path)
    chunks = s.chunks_for_video(vid)
    kinds = {c["kind"] for c in chunks}
    assert kinds == {"transcript", "scene"}
    assert s.get_video(vid)["status"] == "done"

def test_partial_index_when_vision_fails(monkeypatch, tmp_path):
    patch_all(monkeypatch, tmp_path, vision_fails=True)
    s = SqliteStore(tmp_path / "t.db"); vec = FakeVector()
    vid = s.insert_video("a.mp4", str(tmp_path / "a.mp4"), "h1")
    (tmp_path / "a.mp4").write_bytes(b"x")
    pipeline.process_video(vid, s, vec, data_dir=tmp_path)
    chunks = s.chunks_for_video(vid)
    assert any(c["kind"] == "transcript" for c in chunks)  # 자막은 색인됨
    assert s.get_jobs(vid)["vision"]["status"] == "failed"   # vision은 실패 기록
    assert s.get_video(vid)["status"] == "done"              # 부분 완료
```

- [ ] **Step 2: 실패 확인**

Run: `pytest tests/test_pipeline.py -v`
Expected: FAIL

- [ ] **Step 3: 구현**

```python
# archive/pipeline.py
from pathlib import Path
from archive import media, transcribe, embed
from archive.enrich import subtitle, vision
from archive.events import bus, Event

def _emit(video_id, step, status, message=""):
    bus.publish(Event(video_id=video_id, step=step, status=status, message=message))

def _step(store, video_id, step, fn):
    """단계 실행 래퍼: 실패해도 jobs에 기록하고 None 반환(부분 색인 허용)."""
    _emit(video_id, step, "running")
    store.set_job(video_id, step, "running")
    try:
        result = fn()
        store.set_job(video_id, step, "done")
        _emit(video_id, step, "done")
        return result
    except Exception as e:
        store.set_job(video_id, step, "failed", error=str(e))
        _emit(video_id, step, "failed", str(e))
        return None

def process_video(video_id, store, vector_store, data_dir: Path):
    store.set_video_status(video_id, "processing")
    video = store.get_video(video_id)
    src = Path(video["stored_path"])

    audio = _step(store, video_id, "audio",
                  lambda: media.extract_audio(src, data_dir / f"audio_{video_id}.wav"))

    segments = []
    if audio:
        segments = _step(store, video_id, "transcribe",
                         lambda: transcribe.transcribe(audio)) or []
    if segments:
        corrected = _step(store, video_id, "subtitle",
                          lambda: subtitle.correct(segments))
        if corrected:
            segments = corrected

    frames = _step(store, video_id, "keyframes",
                   lambda: media.extract_keyframes(
                       src, data_dir / "frames" / str(video_id),
                       threshold=0.3)) or []

    scenes = _step(store, video_id, "vision",
                   lambda: _analyze_frames(frames)) or []

    # --- 청크 적재 (존재하는 것만; 제약③) ---
    chunk_rows = []
    for i, seg in enumerate(segments):
        cid = store.insert_chunk(video_id, "transcript", i, seg["start"], seg["end"],
                                 seg["text"])
        chunk_rows.append((cid, seg["text"], "transcript", seg["start"], seg["end"], False))
    for i, sc in enumerate(scenes):
        cid = store.insert_chunk(video_id, "scene", i, sc["start"], sc["end"],
                                 sc["description"], frame_path=sc["frame_path"])
        flags = vision.real_flags(sc["flags"])
        for f in flags:
            store.add_flag(cid, f["category"], f.get("severity", "low"), f.get("note"))
        if flags:
            _emit(video_id, "vision", "info", f"금칙 감지: {flags[0]['category']}")
        chunk_rows.append((cid, sc["description"], "scene", sc["start"], sc["end"],
                           bool(flags)))

    # --- 임베딩 + 벡터 색인 ---
    def do_embed():
        texts = [r[1] for r in chunk_rows]
        vectors = embed.embed_texts(texts)
        for (cid, text, kind, st, en, has_flag), v in zip(chunk_rows, vectors):
            vector_store.add(cid, v, text, {"video_id": video_id, "kind": kind,
                             "start_s": st, "end_s": en, "has_flags": has_flag})
            store.mark_embedded(cid)
    if chunk_rows:
        _step(store, video_id, "embed", do_embed)

    store.set_job(video_id, "finalize", "done")
    store.set_video_status(video_id, "done")
    _emit(video_id, "finalize", "done", "처리 완료")

def _analyze_frames(frames):
    out = []
    for i, frame in enumerate(frames):
        res = vision.analyze_frame(frame)
        out.append({"description": res["description"], "flags": res.get("flags", []),
                    "start_s": float(i), "end_s": float(i) + 1.0,
                    "frame_path": str(frame)})
    return out

def resume_incomplete(store, vector_store, data_dir: Path):
    """재시작 시 미완 영상 이어서 처리 (제약①)."""
    for vid in store.incomplete_videos():
        process_video(vid, store, vector_store, data_dir)
```

- [ ] **Step 4: 통과 확인**

Run: `pytest tests/test_pipeline.py -v`
Expected: PASS

- [ ] **Step 5: 커밋**

```bash
git add archive/pipeline.py tests/test_pipeline.py
git commit -m "feat: processing pipeline with checkpoints and partial indexing"
```

---

## Task 14: runbook — 운영 절차서 생성

**Files:**
- Create: `archive/runbook.py`
- Test: `tests/test_runbook.py`

- [ ] **Step 1: 실패 테스트 작성**

```python
# tests/test_runbook.py
from archive.runbook import generate_runbook

def test_runbook_written(tmp_path):
    out = tmp_path / "RUNBOOK.md"
    generate_runbook(out)
    text = out.read_text()
    assert "# 운영 절차서" in text
    assert ".env" in text
    assert "ffmpeg" in text
```

- [ ] **Step 2: 실패 확인**

Run: `pytest tests/test_runbook.py -v`
Expected: FAIL

- [ ] **Step 3: 구현**

```python
# archive/runbook.py
from pathlib import Path

RUNBOOK = """# 운영 절차서 (시스템 런북)

## 1. 사전 요구사항
- Python 3.12+, 시스템 `ffmpeg` 설치 (`ffmpeg -version` 확인)

## 2. 설치
```
python3 -m venv .venv && . .venv/bin/activate
pip install -r requirements.txt
```

## 3. 환경 변수 (.env)
`.env.example`를 복사해 채운다.
- `OPENAI_API_KEY` (필수: STT/교정/Vision/임베딩/쿼리분해)
- `TELEGRAM_BOT_TOKEN`, `TELEGRAM_CHAT_ID` (선택: 알림/봇 검색)
- `DATA_DIR` (기본 ./data), `HYBRID_ALPHA` (기본 0.5), `KEYFRAME_THRESHOLD` (기본 0.3)

## 4. 실행
```
uvicorn archive.web.app:app --reload
```
브라우저에서 http://localhost:8000 접속.

## 5. 데이터 영속
모든 상태는 `DATA_DIR` 아래에 저장된다 (영상/프레임/app.db/chroma).
재시작하면 미완 영상은 자동으로 이어서 처리된다.

## 6. 장애 대응
- 한 단계 실패해도 가능한 범위까지 색인된다. `jobs` 테이블에서 실패 단계 확인.
- 특정 영상 재처리: 해당 영상의 `jobs` 행을 삭제 후 재시작.
- OpenAI 키 누락 시 처리/검색 호출 시점에만 오류가 난다 (지연 초기화).
"""

def generate_runbook(out_path: Path):
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(RUNBOOK, encoding="utf-8")
```

- [ ] **Step 4: 통과 확인**

Run: `pytest tests/test_runbook.py -v`
Expected: PASS

- [ ] **Step 5: 커밋**

```bash
git add archive/runbook.py tests/test_runbook.py
git commit -m "feat: system runbook generator"
```

---

## Task 15: notify/telegram — 알림 + 봇 검색

**Files:**
- Create: `archive/notify/__init__.py` (빈 파일), `archive/notify/telegram.py`
- Test: `tests/test_telegram.py`

httpx로 Telegram Bot API 호출. 테스트는 httpx 호출과 검색 포맷팅을 검증.

- [ ] **Step 1: 실패 테스트 작성**

```python
# tests/test_telegram.py
import archive.notify.telegram as tg

def test_send_message_posts_to_api(monkeypatch):
    captured = {}
    def fake_post(url, json=None, timeout=None):
        captured["url"] = url; captured["json"] = json
        class R:
            def raise_for_status(self): pass
        return R()
    monkeypatch.setattr(tg.httpx, "post", fake_post)
    tg.send_message("hello", token="T", chat_id="C")
    assert "T" in captured["url"] and captured["json"]["text"] == "hello"

def test_send_message_noop_without_token(monkeypatch):
    called = []
    monkeypatch.setattr(tg.httpx, "post", lambda *a, **k: called.append(1))
    tg.send_message("hi", token=None, chat_id=None)
    assert called == []

def test_format_results():
    results = [{"video_id": 1, "kind": "transcript", "start_s": 0.0, "end_s": 2.0,
                "text": "강아지", "explanation": {"text": "키워드 일치 → keyword"}}]
    out = tg.format_results(results)
    assert "강아지" in out and "keyword" in out

def test_event_to_message():
    from archive.events import Event
    msg = tg.event_to_message(Event(1, "transcribe", "done", ""))
    assert "transcribe" in msg and "✅" in msg
```

- [ ] **Step 2: 실패 확인**

Run: `pytest tests/test_telegram.py -v`
Expected: FAIL

- [ ] **Step 3: 구현**

```python
# archive/notify/telegram.py
import httpx
from archive import config

ICON = {"running": "⏳", "done": "✅", "failed": "❌", "skipped": "⏭️", "info": "ℹ️"}

def send_message(text, token=None, chat_id=None):
    token = token or config.telegram_token()
    chat_id = chat_id or config.telegram_chat_id()
    if not token or not chat_id:
        return  # 토큰 없으면 무동작 (웹은 정상)
    url = f"https://api.telegram.org/bot{token}/sendMessage"
    r = httpx.post(url, json={"chat_id": chat_id, "text": text}, timeout=10)
    r.raise_for_status()

def event_to_message(event) -> str:
    icon = ICON.get(event.status, "")
    extra = f" — {event.message}" if event.message else ""
    return f"{icon} [영상 {event.video_id}] {event.step}{extra}"

def format_results(results) -> str:
    if not results:
        return "검색 결과 없음"
    lines = []
    for r in results[:5]:
        ts = f"{r['start_s']:.0f}s"
        lines.append(f"• [영상 {r['video_id']} @{ts}] {r['text'][:60]}\n  ↳ {r['explanation']['text']}")
    return "\n".join(lines)

def subscribe_to_events():
    """이벤트 버스를 구독해 진행 상황을 텔레그램으로 push."""
    from archive.events import bus
    bus.subscribe(lambda e: send_message(event_to_message(e)))
```

- [ ] **Step 4: 통과 확인**

Run: `pytest tests/test_telegram.py -v`
Expected: PASS

- [ ] **Step 5: 봇 폴링 핸들러 추가 (검색 명령)**

`archive/notify/telegram.py`에 추가:
```python
def handle_update(update, engine) -> None:
    """텔레그램 update 1건 처리. /search <쿼리> 명령이면 검색 후 응답."""
    msg = update.get("message", {})
    text = msg.get("text", "")
    chat_id = str(msg.get("chat", {}).get("id", ""))
    if text.startswith("/search"):
        query = text[len("/search"):].strip()
        from archive.search.query import decompose
        parsed = decompose(query)
        results = engine.search(parsed["semantic_query"], mode="hybrid",
                                filters=parsed["filters"])
        send_message(format_results(results), chat_id=chat_id)

async def poll_loop(engine, token=None):
    """getUpdates 롱폴링 루프 (앱 백그라운드 태스크)."""
    import asyncio
    token = token or config.telegram_token()
    if not token:
        return
    offset = 0
    base = f"https://api.telegram.org/bot{token}"
    async with httpx.AsyncClient(timeout=35) as client:
        while True:
            try:
                r = await client.get(f"{base}/getUpdates",
                                     params={"offset": offset, "timeout": 30})
                for upd in r.json().get("result", []):
                    offset = upd["update_id"] + 1
                    handle_update(upd, engine)
            except Exception:
                await asyncio.sleep(3)
```

테스트 `tests/test_telegram.py`에 추가:
```python
def test_handle_update_search(monkeypatch):
    sent = {}
    monkeypatch.setattr(tg, "send_message", lambda text, chat_id=None: sent.update(text=text))
    class Eng:
        def search(self, q, mode, filters):
            return [{"video_id": 1, "kind": "transcript", "start_s": 0.0, "end_s": 1.0,
                     "text": "강아지", "explanation": {"text": "→ both"}}]
    monkeypatch.setattr("archive.search.query.decompose",
                        lambda q: {"semantic_query": q, "keywords": [q], "filters": {}})
    upd = {"message": {"text": "/search 강아지", "chat": {"id": 99}}}
    tg.handle_update(upd, Eng())
    assert "강아지" in sent["text"]
```

- [ ] **Step 6: 통과 확인 + 커밋**

Run: `pytest tests/test_telegram.py -v`
Expected: PASS
```bash
git add archive/notify/__init__.py archive/notify/telegram.py tests/test_telegram.py
git commit -m "feat: telegram notifications and /search bot command"
```

---

## Task 16: web/app — FastAPI (업로드·SSE·검색)

**Files:**
- Create: `archive/web/__init__.py` (빈 파일), `archive/web/app.py`, `archive/web/static/index.html`
- Test: `tests/test_web.py`

`app.py`는 앱 상태(store/vector/engine)를 모듈 싱글톤으로 구성하되, 테스트에서 주입 교체 가능하게 팩토리로 분리.

- [ ] **Step 1: 실패 테스트 작성 (TestClient)**

```python
# tests/test_web.py
from fastapi.testclient import TestClient
from archive.web import app as webapp
from archive.store.sqlite_store import SqliteStore

class FakeVector:
    def add(self, *a, **k): pass
    def query(self, *a, **k): return []

def client(tmp_path, monkeypatch):
    store = SqliteStore(tmp_path / "t.db")
    monkeypatch.setattr(webapp, "get_store", lambda: store)
    monkeypatch.setattr(webapp, "get_vector", lambda: FakeVector())
    monkeypatch.setattr(webapp, "schedule_processing", lambda vid: None)
    return TestClient(webapp.app), store

def test_upload_creates_video(tmp_path, monkeypatch):
    c, store = client(tmp_path, monkeypatch)
    r = c.post("/upload", files={"file": ("a.mp4", b"data", "video/mp4")})
    assert r.status_code == 200
    assert "video_id" in r.json()
    assert len(store.list_videos()) == 1

def test_search_returns_json(tmp_path, monkeypatch):
    c, store = client(tmp_path, monkeypatch)
    vid = store.insert_video("a.mp4", "/x", "h1")
    store.insert_chunk(vid, "transcript", 0, 0.0, 1.0, "강아지 공원")
    r = c.get("/search", params={"q": "강아지", "mode": "keyword"})
    assert r.status_code == 200
    data = r.json()
    assert data[0]["text"].startswith("강아지")
    assert data[0]["explanation"]["source"] == "keyword"

def test_index_page_served(tmp_path, monkeypatch):
    c, _ = client(tmp_path, monkeypatch)
    r = c.get("/")
    assert r.status_code == 200 and "업로드" in r.text
```

- [ ] **Step 2: 실패 확인**

Run: `pytest tests/test_web.py -v`
Expected: FAIL

- [ ] **Step 3: 구현 (app.py)**

```python
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
```

- [ ] **Step 4: 프론트 index.html 작성**

```html
<!-- archive/web/static/index.html -->
<!doctype html><html lang="ko"><head><meta charset="utf-8">
<title>영상 아카이브</title>
<style>
 body{font-family:system-ui;max-width:760px;margin:2rem auto;padding:0 1rem}
 .badge{font-size:.75rem;padding:.1rem .4rem;border-radius:.3rem;color:#fff}
 .both{background:#7c3aed}.keyword{background:#2563eb}.vector{background:#059669}
 .filter{background:#64748b}
 .result{border:1px solid #e2e8f0;border-radius:.5rem;padding:.8rem;margin:.5rem 0}
 .expl{color:#64748b;font-size:.85rem;margin-top:.3rem}
 progress{width:100%}
</style></head><body>
<h1>영상 아카이브 업로드</h1>
<input type="file" id="file" accept="video/*">
<button onclick="upload()">업로드</button>
<progress id="bar" max="10" value="0" hidden></progress>
<div id="status"></div>
<hr>
<h2>검색</h2>
<input id="q" placeholder="자연어로 검색..." size="40">
<select id="mode"><option>hybrid</option><option>keyword</option>
 <option>vector</option><option>filter</option></select>
<button onclick="search()">검색</button>
<div id="results"></div>
<script>
const STEPS=["audio","transcribe","subtitle","keyframes","vision","embed","finalize"];
async function upload(){
 const f=document.getElementById('file').files[0]; if(!f)return;
 const fd=new FormData(); fd.append('file',f);
 const r=await fetch('/upload',{method:'POST',body:fd});
 const {video_id}=await r.json();
 const bar=document.getElementById('bar'); bar.hidden=false; bar.value=0;
 const es=new EventSource(`/progress/${video_id}`);
 es.onmessage=e=>{const d=JSON.parse(e.data);
  bar.value=STEPS.indexOf(d.step)+1;
  document.getElementById('status').textContent=`${d.step}: ${d.status} ${d.message||''}`;
  if(d.step==='finalize'&&d.status==='done')es.close();};
}
async function search(){
 const q=document.getElementById('q').value, mode=document.getElementById('mode').value;
 const r=await fetch(`/search?q=${encodeURIComponent(q)}&mode=${mode}`);
 const data=await r.json();
 document.getElementById('results').innerHTML=data.map(x=>`
  <div class="result"><span class="badge ${x.source}">${x.source}</span>
   <b> [영상 ${x.video_id} @${x.start_s.toFixed(0)}s]</b> ${x.text}
   <div class="expl">↳ ${x.explanation.text}</div></div>`).join('')||'결과 없음';
}
</script></body></html>
```

- [ ] **Step 5: 통과 확인**

Run: `pytest tests/test_web.py -v`
Expected: PASS

- [ ] **Step 6: 커밋**

```bash
git add archive/web/__init__.py archive/web/app.py archive/web/static/index.html tests/test_web.py
git commit -m "feat: FastAPI web app with upload, SSE progress, search UI"
```

---

## Task 17: 전체 검증 + 런북 생성 + README

**Files:**
- Create: `README.md`
- Modify: 없음 (검증 단계)

- [ ] **Step 1: 전체 테스트 실행**

Run: `pytest -v`
Expected: ALL PASS

- [ ] **Step 2: 런북 생성 스모크**

Run:
```bash
python -c "from archive.runbook import generate_runbook; from pathlib import Path; generate_runbook(Path('docs/RUNBOOK.md'))"
```
Expected: `docs/RUNBOOK.md` 생성됨

- [ ] **Step 3: 서버 기동 스모크 (수동)**

Run: `uvicorn archive.web.app:app` → http://localhost:8000 접속 → 업로드 UI 표시 확인.
(OPENAI_API_KEY가 있어야 실제 처리됨. 없으면 업로드는 되고 처리 단계에서 jobs에 실패 기록.)

- [ ] **Step 4: README 작성**

```markdown
# 영상 아카이브 & 설명 가능한 검색

영상을 업로드하면 Whisper STT + GPT Vision으로 색인하고, 키워드(FTS5)+의미(임베딩)
하이브리드 검색에 출처·이유 설명을 붙여 반환하는 영속 아카이브.

## 빠른 시작
1. `cp .env.example .env` 후 `OPENAI_API_KEY` 입력
2. `pip install -r requirements.txt` (시스템 `ffmpeg` 필요)
3. `uvicorn archive.web.app:app --reload`
4. http://localhost:8000

자세한 운영은 `docs/RUNBOOK.md` 참고.
```

- [ ] **Step 5: 커밋**

```bash
git add README.md docs/RUNBOOK.md
git commit -m "docs: README and generated runbook"
```

---

## 완료 기준

- `pytest -v` 전부 통과
- 업로드 → 단계별 SSE 진행 → 색인 완료까지 동작
- 4모드 검색 모두 출처/설명 반환
- 재시작 후 검색 결과 유지 (SQLite + Chroma 영속)
- 한 단계 실패해도 가능한 범위까지 색인 (pipeline 부분 색인 테스트로 보장)
- OpenAI 키 누락 시 사용 시점에만 오류 (지연 초기화)
- 텔레그램 토큰 설정 시 진행 알림 + `/search` 봇 검색 동작
