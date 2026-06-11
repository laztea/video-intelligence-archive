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
    """이벤트 버스를 구독해 영상 처리 '완료'만 텔레그램으로 push.

    단계별 진행은 웹 프로그래스 바(SSE)로 확인하고, 텔레그램에는
    한 영상당 완료 메시지 한 건만 보낸다 (알림 과다 방지)."""
    from archive.events import bus
    def on_event(e):
        if e.step == "finalize" and e.status == "done":
            # 파이프라인이 만든 요약(작업 내역·금칙·실패) 메시지를 그대로 전송
            send_message(e.message or f"✅ [영상 {e.video_id}] 처리 완료")
    bus.subscribe(on_event)

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
