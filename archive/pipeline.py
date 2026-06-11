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
