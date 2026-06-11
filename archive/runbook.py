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
