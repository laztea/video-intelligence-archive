# archive/enrich/vision.py
import base64
import json
from pathlib import Path
from archive import config

MODEL = "gpt-5.5"

SYSTEM = ("너는 영상 프레임 분석기다. 이미지를 보고 (1) 장면을 한국어로 1~2문장 묘사하고 "
          "(2) 금칙 요소(선정성 sexual, 폭력 violence, 혐오 hate, 로고/저작권 logo)를 "
          'category/severity(low|medium|high)/note 로 태깅하라. 해당 없으면 category="none". '
          "참고 대사가 함께 주어지면, 이미지만으로 모호한 장소·상황(예: 실내가 학교인지 사무실인지, "
          "관계, 사건)을 대사 맥락으로 추론해 묘사에 반영하라. 단, 대사에 근거가 없으면 단정하지 마라. "
          '반드시 {"description": str, "flags": [{"category","severity","note"}]} JSON으로만 답하라.')

def analyze_frame(frame_path: Path, context: str = "") -> dict:
    client = config.get_openai()
    b64 = base64.b64encode(Path(frame_path).read_bytes()).decode()
    prompt = "이 프레임을 분석하라."
    if context:
        prompt += f"\n\n[이 시점 전후의 대사 — 장소·상황 추론에 참고]\n{context}"
    resp = client.chat.completions.create(
        model=MODEL,
        response_format={"type": "json_object"},
        messages=[
            {"role": "system", "content": SYSTEM},
            {"role": "user", "content": [
                {"type": "text", "text": prompt},
                {"type": "image_url",
                 "image_url": {"url": f"data:image/jpeg;base64,{b64}"}}]}])
    data = json.loads(resp.choices[0].message.content)
    data.setdefault("flags", [])
    return data

def real_flags(flags: list[dict]) -> list[dict]:
    """category='none' 제외한 실제 금칙만."""
    return [f for f in flags if f.get("category") and f["category"] != "none"]
