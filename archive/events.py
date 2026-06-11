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
