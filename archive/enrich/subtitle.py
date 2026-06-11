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
