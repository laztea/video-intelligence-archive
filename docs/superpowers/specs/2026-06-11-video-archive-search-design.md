# 멀티모달 영상 아카이브 & 설명 가능한 검색 — 설계 문서

작성일: 2026-06-11

## 1. 개요

영상을 업로드하면 AI가 듣고(STT)·보고(Vision)·정리해서, 나중에 자연어로 찾을 수 있는
개인용 **멀티모달 영상 아카이브 + 설명 가능한 검색(explainable search)** 서비스.

핵심 차별점: 검색 결과마다 **출처(keyword/vector/both)** 와 **왜 나왔는지 설명**을 함께 반환.

### 확정 결정 요약

| 항목 | 결정 |
|---|---|
| 스택 | Python + FastAPI + 경량 HTML/JS 프론트 |
| 트랜스크립트 | Whisper STT (OpenAI Audio API, 로컬 whisper.cpp는 대안) |
| 처리 | 백그라운드 잡 + SSE 진행률 |
| 프레임 | ffmpeg keyframe(장면 전환) 추출 |
| 금칙 검수 | 태그만 달고 아카이브 (차단 없음) |
| 운영절차서 | 시스템 런북/세팅 `.md` 자동 생성/갱신 |
| 텔레그램 | 단방향 알림 + 봇 검색(양방향) |
| 검색 단위 | 타임스탬프 청크 (자막 세그먼트 / 장면) |
| 설명 생성 | 결정론적 조립 (호출 비용 0) |
| AI 모델 | GPT-5.5 (교정·Vision·쿼리분해), text-embedding-3-small |

### 제약 (스펙 명시)

1. 결과는 재시작해도 유지 — SQLite·ChromaDB 모두 파일 영속
2. API 키·토큰은 `.env`
3. 한 단계 실패해도 가능한 범위까지 색인 — 단계별 체크포인트·부분 색인
4. GPT/임베딩 호출 키는 사용 시점에 지연 초기화

---

## 2. 전체 구조 & 모듈 분해

각 모듈은 단일 책임 · 명확한 인터페이스 · 독립 테스트 가능.
검색 엔진은 웹과 텔레그램 봇이 **공유**하는 모듈.

```
archive/
├── config.py          # .env 로드 + 지연 초기화 팩토리 (get_openai/get_chroma/get_db/get_telegram)
├── events.py          # 인프로세스 진행 이벤트 버스 → SSE + 텔레그램 팬아웃
│
├── media.py           # ffmpeg 래퍼: 오디오 추출 · keyframe(장면전환) 추출
├── transcribe.py      # Whisper STT → [{start,end,text}] 세그먼트
├── enrich/
│   ├── subtitle.py    # GPT-5.5 자막 교정
│   └── vision.py      # GPT-5.5 Vision: 장면 분석 + 금칙 태깅 (keyframe 단위)
├── embed.py           # text-embedding-3-small 배치 임베딩
│
├── store/
│   ├── sqlite_store.py  # 테이블 + FTS5 (영속)
│   └── vector_store.py  # ChromaDB (영속)
│
├── pipeline.py        # 영상 1개 처리 오케스트레이션 (단계별 체크포인트·부분색인·재개)
├── runbook.py         # 운영 절차서(.md) 생성/갱신
│
├── search/
│   ├── query.py       # GPT-5.5 자연어 쿼리 분해 → {keywords, filters, semantic_query}
│   ├── engine.py      # 4모드 하이브리드 (FTS5+벡터 점수 합산)
│   └── explain.py     # 결정론적 출처/이유 조립
│
├── notify/telegram.py # 알림 push + 봇 명령(검색) 핸들러
└── web/
    ├── app.py         # FastAPI: 업로드 · SSE 진행률 · 검색 API
    └── static/        # 업로드/검색 UI
```

의존 방향: `web`·`notify` → `pipeline`·`search` → `store`·`enrich`·`embed`·`media` → `config`·`events`.
하위는 상위를 모름 → 테스트 시 GPT/Chroma/Whisper/ffmpeg를 목으로 대체 가능.

---

## 3. 처리 파이프라인 (`pipeline.py`)

영상 1개당 순차 단계. **각 단계는 독립 체크포인트** — try/except로 감싸 실패해도 `jobs`에
기록 후 가능한 다음 단계로 진행. 각 단계 시작/완료/실패는 `events.py`로 발행 →
SSE(프로그래스 바) + 텔레그램 동시 전파.

```
 1. persist     업로드 파일 저장 + SHA256 해시 (중복 스킵)
 2. audio       ffmpeg 오디오 추출
 3. transcribe  Whisper → 자막 세그먼트 (transcript 청크)
 4. subtitle    GPT 자막 교정          (실패 시 원본 세그먼트 유지)
 5. keyframes   ffmpeg 장면전환 프레임 추출
 6. vision      GPT Vision 장면분석 + 금칙 태깅 (scene 청크)  (실패 시 해당 프레임 스킵)
 7. embed       모든 청크 임베딩 생성   (실패한 청크는 임베딩 없이 키워드 색인만)
 8. index-sql   SQLite 테이블 + FTS5 기록
 9. index-vec   ChromaDB 기록 (임베딩 있는 청크만)
10. finalize    상태=done, 런북 갱신, 텔레그램 완료 알림
```

- 부분 색인(제약③): 4·6·7 실패 시에도 **이미 만들어진 청크는 8~9에서 색인**.
- 재개: 재시작 시 `jobs` 상태로 미완 영상의 마지막 성공 단계 다음부터 이어 처리.
- 청크 종류: `transcript`(자막 세그먼트, 타임스탬프) / `scene`(keyframe, start~end + Vision 설명 + 금칙 플래그).

---

## 4. 데이터 모델

### 저장 레이아웃 (모두 `./data`, gitignore)

```
data/
├── uploads/<sha256>.<ext>      # 원본 영상 (해시 파일명, 중복 제거)
├── frames/<video_id>/<seq>.jpg # 추출 keyframe
├── app.db                       # SQLite
└── chroma/                      # ChromaDB 영속
docs/RUNBOOK.md                  # 운영 절차서
```

### SQLite 스키마

```sql
videos(
  id INTEGER PK, filename TEXT, stored_path TEXT, sha256 TEXT UNIQUE,
  duration_s REAL, status TEXT,         -- pending|processing|done|failed
  error TEXT, created_at TEXT, updated_at TEXT
)

chunks(
  id INTEGER PK, video_id INTEGER FK, kind TEXT,  -- transcript|scene
  seq INTEGER, start_s REAL, end_s REAL,
  text TEXT, frame_path TEXT,                      -- scene만
  embedded INTEGER DEFAULT 0, created_at TEXT
)

chunk_flags(                                       -- 금칙 태그
  id INTEGER PK, chunk_id INTEGER FK,
  category TEXT, severity TEXT, note TEXT
)

jobs(                                              -- 단계별 체크포인트
  id INTEGER PK, video_id INTEGER FK, step TEXT,
  status TEXT,                                     -- pending|running|done|failed|skipped
  error TEXT, started_at TEXT, finished_at TEXT
)

chunks_fts  -- FTS5(text, content='chunks', content_rowid='id') + 동기화 트리거
```

### ChromaDB

- 영속 클라이언트: `./data/chroma`
- collection `chunks`: `ids=str(chunk_id)`, `embeddings`,
  `metadatas={video_id, kind, start_s, end_s, has_flags}`, `documents=text`

---

## 5. 하이브리드 검색 & 설명 (`search/`)

### 쿼리 분해 (`query.py`, GPT)

자연어 쿼리 → 구조화 JSON `{keywords:[], semantic_query:str, filters:{video?, kind?, time?, flagged?}}`.
GPT 실패 시 폴백: 원본 쿼리를 keyword·vector 양쪽에 그대로 사용 (제약③ 정신).

### 4 모드 (`engine.py`)

| 모드 | 동작 |
|---|---|
| `keyword` | FTS5 `MATCH` → `bm25()` 랭킹 (낮을수록 좋음 → 반전·정규화) |
| `vector` | 쿼리 임베딩 → Chroma 유사도 → `sim = 1/(1+distance)` 정규화 |
| `hybrid` | `score = α·vec_norm + (1-α)·kw_norm`, 한쪽만 매칭 시 결측=0, α 설정값(기본 0.5) |
| `filter` | 점수 없이 메타데이터 필터만 (video/kind/시간/flagged), 시간순 정렬 |

- 결과는 `chunk_id`로 dedup, 결과집합 내 min-max 정규화 후 가중합.

### 설명 (`explain.py`, 결정론적)

각 결과에 대해:
- `source`: `keyword` | `vector` | `both`
- `detail`: 매칭된 키워드 텀, bm25 점수, 벡터 유사도, 적용된 필터
- 사람용 문자열 예: `"키워드 '강아지' 일치 (FTS 3.2) + 의미 유사도 0.81 → both"`

반환 객체 (웹·텔레그램 공통):
```
{ video_id, filename, kind, start_s, end_s, text, flags, score, source, explanation }
```

---

## 6. 웹 & 텔레그램 인터페이스

### 웹 (FastAPI, `web/app.py`)

| 엔드포인트 | 역할 |
|---|---|
| `GET /` | 업로드 + 검색 단일 페이지 UI |
| `POST /upload` | 파일 저장 → video/jobs 생성 → 백그라운드 처리 시작 → `{video_id}` 반환 |
| `GET /progress/{video_id}` | SSE 스트림 (단계별 진행) |
| `GET /search` | `q, mode, filters` → JSON 결과(설명 포함) |
| `GET /video/{id}` | 영상 메타 + 청크 |

- 백그라운드: FastAPI BackgroundTasks/asyncio. 시작 시 `jobs` 기준 미완 영상 자동 재개.
- 프론트: 업로드 프로그래스 바(SSE), 검색창 + 모드 선택, 결과에 텍스트·타임스탬프·**출처 배지**·**설명** 표시.

### 텔레그램 (`notify/telegram.py`)

- Push: 이벤트 버스 구독 → 단계 완료/실패·금칙 감지·영상 완료 시 메시지.
- 봇 명령: `/search <쿼리> [mode]` → `search.engine` 호출 → 상위 N개 + 설명 반환. `/status`, `/list`.
- long polling 기반. 토큰 지연 초기화(.env). 토큰 없으면 봇 비활성(웹은 정상 동작).

---

## 7. 에러 처리 · 영속 · 설정

- `config.py`: `python-dotenv`로 `.env` 로드. **지연 초기화 캐시** — `get_openai()`/`get_chroma()`/
  `get_db()`/`get_telegram()`가 첫 호출 시 생성. 키 없으면 **그 기능 사용 시점에만** 예외 (제약④).
- 영속(제약①): 모든 상태가 `./data` 파일 — SQLite + Chroma 파일 영속. 재시작 시 `jobs`로 재개.
- 부분 실패(제약③): 단계별 try/except → `jobs` 기록 → 진행. 색인 단계는 존재하는 청크만 대상.
- 멱등성: SHA256 dedup. 재처리 시 `jobs` 체크포인트로 완료 단계 스킵.
- 시크릿(제약②): `.env`에만. `.gitignore`에 `.env`, `data/` 포함.

---

## 8. 테스트 전략

- 모듈별 단위 테스트, 외부 의존(OpenAI/Chroma/ffmpeg/Whisper)은 목으로 대체.
- `store`: 임시 SQLite + 임시 Chroma로 FTS5·벡터 라운드트립 검증.
- `search/engine`: 알려진 청크 시드 → 점수·정규화·모드별 동작·dedup 검증.
- `search/explain`: 결정론적 → 알려진 입력에 대한 정확한 설명 문자열 단언.
- `pipeline`: 단계 실패 주입 → 부분 색인 + `jobs` 기록 검증.
- 통합: 작은 픽스처 영상 → GPT/Whisper 목으로 파이프라인 실행 → 검색 결과 검증.
- 가능한 곳은 TDD.

### 주요 의존성

`fastapi`, `uvicorn`, `openai`, `chromadb`, `python-dotenv`, `httpx`(텔레그램), `sse-starlette`, `pytest` + 시스템 `ffmpeg`.

---

## 9. 빌드 순서 (구현 계획 입력)

1. `config` + `events` + `store`(SQLite/FTS5, Chroma) — 토대
2. `media` + `transcribe` + `embed` — 추출/임베딩
3. `enrich`(subtitle/vision) — GPT 강화
4. `pipeline` — 단계 오케스트레이션·부분색인·재개
5. `search`(query/engine/explain) — 4모드·설명
6. `web`(app + static) — 업로드·SSE·검색 UI
7. `notify/telegram` — 알림 + 봇 검색
8. `runbook` — 운영 절차서 생성
