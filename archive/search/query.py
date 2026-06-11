# archive/search/query.py
import json
from archive import config

MODEL = "gpt-5.5"

SYSTEM = ("너는 검색 쿼리 분해기다. 사용자의 자연어 검색을 "
          '{"keywords": [...], "semantic_query": "의미 검색용 문장", '
          '"filters": {"kind": "transcript|scene", "flagged": true}} 형태 JSON으로 분해하라. '
          "keywords에는 핵심 명사와 함께 **동의어·유의어**를 포함하라 "
          "(예: '오피스'→[\"오피스\",\"사무실\"], '차'→[\"차\",\"자동차\",\"승용차\"]). "
          "특히 한국어 고유어와 외래어 짝(사무실↔오피스, 모임↔미팅 등)을 함께 넣어라. "
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
