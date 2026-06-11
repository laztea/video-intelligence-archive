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
  text, content='chunks', content_rowid='id', tokenize='trigram');

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
