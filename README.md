# 영상 아카이브 & 설명 가능한 검색

영상을 업로드하면 Whisper STT + GPT Vision으로 색인하고, 키워드(FTS5)+의미(임베딩)
하이브리드 검색에 출처·이유 설명을 붙여 반환하는 영속 아카이브.

## 빠른 시작
1. `cp .env.example .env` 후 `OPENAI_API_KEY` 입력
2. `pip install -r requirements.txt` (시스템 `ffmpeg` 필요)
3. `uvicorn archive.web.app:app --reload`
4. http://localhost:8000

자세한 운영은 `docs/RUNBOOK.md` 참고.
