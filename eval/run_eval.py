"""영상 장면 벡터 검색 품질 자동 평가 파이프라인 (사람 개입 없이 끝까지 실행).

설계 — known-item 평가:
  코퍼스의 각 장면 설명으로부터 "그 장면을 찾으려는 사용자 검색어"를 LLM이 생성한다.
  생성된 쿼리의 정답(gold)은 곧 출처 장면이다. 따라서 사람 라벨링 없이도
  Recall@5(정답이 top5에 있나)·MRR(정답 순위)를 객관적으로 계산할 수 있고,
  Precision@5는 LLM-as-judge로 각 결과의 관련성을 판정해 측정한다.

[컨텍스트]
  - 벡터 DB: ChromaDB (영속, ./data/chroma)
  - 임베딩 모델: OpenAI text-embedding-3-small
  - 장면 메타데이터: chunk_id(scene_id) · text(description) · start_s(timestamp) · flags(tags)

실행:  python eval/run_eval.py        (기본 100개)
       EVAL_N=20 python eval/run_eval.py
출력:  eval/report.html · eval/results.json
"""
import os
import sys
import json
import html
import time
import random
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from dotenv import load_dotenv
load_dotenv()

from archive import config, embed
from archive.store.sqlite_store import SqliteStore
from archive.store.vector_store import VectorStore

MODEL = "gpt-5.5"
K = 5
N_QUERIES = int(os.environ.get("EVAL_N", "100"))
OUT = Path(__file__).resolve().parent


def log(m):
    print(f"[eval] {m}", flush=True)


def gpt_json(client, system, user, retries=4):
    """JSON 응답 GPT 호출 (재시도 포함). 끝까지 자동 진행을 위해 견고하게."""
    last = None
    for i in range(retries):
        try:
            r = client.chat.completions.create(
                model=MODEL, response_format={"type": "json_object"},
                messages=[{"role": "system", "content": system},
                          {"role": "user", "content": user}])
            return json.loads(r.choices[0].message.content)
        except Exception as e:  # noqa
            last = e
            time.sleep(1.5 * (i + 1))
    log(f"GPT 호출 실패(스킵): {last}")
    return {}


def load_scenes(store):
    rows = store.conn.execute(
        "SELECT id, video_id, start_s, text FROM chunks "
        "WHERE kind='scene' AND text IS NOT NULL AND text<>''").fetchall()
    return [dict(r) for r in rows]


def gen_queries(client, scenes):
    """각 장면에 대해 사용자가 입력할 법한 자연어 검색어 1개 생성 (10개씩 배치)."""
    out = []
    BATCH = 10
    sys_p = ("너는 영상 장면 검색 평가용 쿼리 생성기다. 각 장면 설명에 대해, 그 장면을 찾으려는 "
             "사용자가 검색창에 입력할 법한 자연스러운 한국어 검색어를 1개씩 만들어라. "
             "설명을 그대로 베끼지 말고 핵심 요소(인물·장소·행동·사물)를 2~8단어로 짧게 표현하라. "
             '출력: {"queries":[{"id":장면id,"query":"검색어"}]} JSON만.')
    for i in range(0, len(scenes), BATCH):
        batch = scenes[i:i + BATCH]
        items = [{"id": s["id"], "description": s["text"]} for s in batch]
        data = gpt_json(client, sys_p, json.dumps({"scenes": items}, ensure_ascii=False))
        byid = {s["id"]: s for s in batch}
        for q in data.get("queries", []):
            sid = q.get("id")
            if sid in byid and q.get("query"):
                out.append({"query": q["query"].strip(), "source_id": sid,
                            "description": byid[sid]["text"]})
        log(f"쿼리 생성 {len(out)}/{min(len(scenes), N_QUERIES)}")
        if len(out) >= N_QUERIES:
            break
    return out[:N_QUERIES]


def judge(client, query, results):
    """top-k 결과 각각의 관련성 판정 (1=관련, 0=무관)."""
    items = [{"i": idx, "description": r.get("text", "")} for idx, r in enumerate(results)]
    sys_p = ("너는 검색 관련성 평가자다. 사용자 검색어와 각 장면 설명을 보고 그 장면이 검색 의도에 "
             "관련 있으면 1, 없으면 0으로 판정하라. "
             '출력: {"judgments":[{"i":인덱스,"relevant":0|1}]} JSON만.')
    data = gpt_json(client, sys_p, json.dumps({"query": query, "scenes": items}, ensure_ascii=False))
    rel = {j["i"]: int(j.get("relevant", 0)) for j in data.get("judgments", []) if "i" in j}
    return [rel.get(idx, 0) for idx in range(len(results))]


def analyze_failures(client, failures):
    """실패(정답이 top5에 없음) 쿼리들의 원인 패턴 분류."""
    if not failures:
        return []
    items = [{"query": f["query"], "source_description": f["description"][:220]}
             for f in failures[:50]]
    sys_p = ("너는 검색 실패 원인 분석가다. 각 (검색어, 정답 장면 설명) 쌍에서 벡터 검색이 정답을 "
             "top5 안에 못 넣은 이유를 다음 중 하나로 분류하라: "
             "'쿼리_과도하게_일반적', '설명_정보부족', '유의어_불일치', '시각정보_텍스트부재', "
             "'중복_유사장면', '기타'. "
             '출력: {"analysis":[{"query":검색어,"category":카테고리,"reason":"한줄 이유"}]} JSON만.')
    data = gpt_json(client, sys_p, json.dumps({"cases": items}, ensure_ascii=False))
    return data.get("analysis", [])


def evaluate():
    store = SqliteStore(config.data_dir() / "app.db")
    vec = VectorStore(config.get_chroma())
    client = config.get_openai()

    scenes = load_scenes(store)
    log(f"장면 코퍼스: {len(scenes)}개")
    if not scenes:
        log("장면 청크가 없습니다. 영상을 먼저 색인하세요. 종료.")
        return None

    random.seed(42)
    random.shuffle(scenes)
    sample = scenes[:N_QUERIES]
    queries = gen_queries(client, sample)
    log(f"골드 쿼리 {len(queries)}개 확보")

    rows = []
    for n, qd in enumerate(queries):
        try:
            hits = vec.query(embed.embed_one(qd["query"]), k=K)
        except Exception as e:  # noqa
            log(f"검색 실패 '{qd['query']}': {e}")
            continue
        for h in hits:
            c = store.get_chunk(h["chunk_id"])
            h["text"] = c["text"] if c else ""
        ids = [h["chunk_id"] for h in hits]
        rank = ids.index(qd["source_id"]) + 1 if qd["source_id"] in ids else None
        rels = judge(client, qd["query"], hits)
        rows.append({
            "query": qd["query"], "source_id": qd["source_id"],
            "description": qd["description"],
            "hits": [{"id": h["chunk_id"], "text": h["text"],
                      "sim": round(1.0 / (1.0 + h["distance"]), 3)} for h in hits],
            "rank": rank, "rels": rels, "hit": rank is not None,
            "rr": (1.0 / rank if rank else 0.0), "p_at_5": sum(rels) / K,
        })
        if (n + 1) % 10 == 0:
            log(f"평가 진행 {n + 1}/{len(queries)}")

    nq = len(rows) or 1
    recall5 = sum(r["hit"] for r in rows) / nq
    prec5 = sum(r["p_at_5"] for r in rows) / nq
    mrr = sum(r["rr"] for r in rows) / nq
    failures = [r for r in rows if not r["hit"]]
    fa = analyze_failures(client, failures)

    metrics = {"n": len(rows), "recall_at_5": recall5, "precision_at_5": prec5,
               "mrr": mrr, "misses": len(failures)}
    log(f"== Recall@5={recall5:.3f}  Precision@5={prec5:.3f}  MRR={mrr:.3f}  "
        f"misses={len(failures)}/{len(rows)} ==")

    result = {"metrics": metrics, "rows": rows, "failure_analysis": fa}
    (OUT / "results.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    write_html(result)
    log(f"리포트 작성 완료: {OUT / 'report.html'}")
    return result


# --------------------------- HTML 리포트 ---------------------------

def esc(t):
    return html.escape(str(t if t is not None else ""))


def write_html(result):
    m = result["metrics"]
    rows = result["rows"]
    fa = result["failure_analysis"]

    # 실패 카테고리 집계
    cats = {}
    for a in fa:
        c = a.get("category", "기타")
        cats[c] = cats.get(c, 0) + 1
    cat_bars = ""
    total_fa = sum(cats.values()) or 1
    for c, n in sorted(cats.items(), key=lambda x: -x[1]):
        pct = n / total_fa * 100
        cat_bars += (f'<div class="fbar"><div class="flabel">{esc(c)}</div>'
                     f'<div class="ftrack"><i style="width:{pct:.0f}%"></i></div>'
                     f'<div class="fn">{n}</div></div>')

    def metric_card(label, val, sub, cls):
        return (f'<div class="mcard {cls}"><div class="mval">{val}</div>'
                f'<div class="mlabel">{label}</div><div class="msub">{sub}</div></div>')

    cards = (
        metric_card("Recall@5", f"{m['recall_at_5']*100:.1f}<span>%</span>",
                    "정답 장면이 top-5에 포함된 비율 (known-item)", "amber") +
        metric_card("Precision@5", f"{m['precision_at_5']*100:.1f}<span>%</span>",
                    "top-5 결과 중 관련 있는 비율 (LLM 판정)", "teal") +
        metric_card("MRR", f"{m['mrr']:.3f}",
                    "정답 장면의 평균 역순위 (1/rank)", "rose") +
        metric_card("쿼리 수", f"{m['n']}",
                    f"실패(top-5 미포함) {m['misses']}건", "slate")
    )

    # per-query rows
    qrows = ""
    for r in rows:
        rank = r["rank"]
        rankbadge = (f'<span class="rk hit">#{rank}</span>' if rank
                     else '<span class="rk miss">miss</span>')
        hits_html = ""
        for j, h in enumerate(r["hits"]):
            relevant = r["rels"][j] if j < len(r["rels"]) else 0
            is_src = (h["id"] == r["source_id"])
            cls = "src" if is_src else ("rel" if relevant else "")
            mark = "🎯" if is_src else ("✓" if relevant else "·")
            hits_html += (f'<div class="hit {cls}"><span class="hm">{mark}</span>'
                          f'<span class="hsim">{h["sim"]}</span>'
                          f'<span class="htx">{esc(h["text"][:90])}</span></div>')
        qrows += (
            f'<tr><td class="q">{esc(r["query"])}{rankbadge}'
            f'<div class="qsrc">정답 장면 #{r["source_id"]} · {esc(r["description"][:80])}…</div></td>'
            f'<td class="hits">{hits_html}'
            f'<div class="pp">P@5 = {r["p_at_5"]:.1f}</div></td></tr>')

    # 실패 케이스 상세
    fa_rows = ""
    for a in fa:
        fa_rows += (f'<tr><td class="fcat">{esc(a.get("category","기타"))}</td>'
                    f'<td>{esc(a.get("query",""))}</td>'
                    f'<td class="freason">{esc(a.get("reason",""))}</td></tr>')

    doc = f"""<!doctype html><html lang="ko"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>검색 품질 평가 리포트</title>
<link rel="preconnect" href="https://fonts.googleapis.com">
<link href="https://fonts.googleapis.com/css2?family=Fraunces:ital,opsz,wght@0,9..144,400;0,9..144,600;0,9..144,900;1,9..144,500&family=Spline+Sans:wght@400;500;600&family=JetBrains+Mono:wght@400;500;700&display=swap" rel="stylesheet">
<style>
:root{{--bg:#14110d;--surface:#1f1a13;--surface2:#262019;--line:#352b1e;--line2:#473a28;
--ink:#ece3d4;--muted:#a2937b;--faint:#6f6353;--amber:#e6a73f;--teal:#5fb8a8;--rose:#d98aa6;--slate:#8aa0b4;--flag:#df6a4a;}}
*{{box-sizing:border-box}}
body{{margin:0;background:radial-gradient(120% 80% at 85% -10%,rgba(230,167,63,.10),transparent 55%),var(--bg);
color:var(--ink);font-family:"Spline Sans",sans-serif;font-size:15px;line-height:1.55}}
.wrap{{max-width:1100px;margin:0 auto;padding:46px 30px 90px}}
.head .eyebrow{{font-family:"JetBrains Mono",monospace;font-size:11px;letter-spacing:.3em;text-transform:uppercase;color:var(--amber)}}
h1{{font-family:"Fraunces",serif;font-weight:900;font-size:42px;letter-spacing:-.02em;margin:.25em 0 .1em}}
h1 em{{font-style:italic;color:var(--amber);font-weight:500}}
.sub{{color:var(--muted);font-size:14.5px;max-width:680px}}
.sub code{{font-family:"JetBrains Mono",monospace;color:var(--teal);font-size:12.5px}}
.metrics{{display:grid;grid-template-columns:repeat(4,1fr);gap:16px;margin:38px 0 14px}}
.mcard{{background:var(--surface);border:1px solid var(--line);border-radius:14px;padding:20px 18px;border-top:3px solid var(--line2)}}
.mcard.amber{{border-top-color:var(--amber)}}.mcard.teal{{border-top-color:var(--teal)}}
.mcard.rose{{border-top-color:var(--rose)}}.mcard.slate{{border-top-color:var(--slate)}}
.mval{{font-family:"Fraunces",serif;font-weight:600;font-size:38px;line-height:1}}
.mval span{{font-size:18px;color:var(--muted)}}
.mlabel{{font-family:"JetBrains Mono",monospace;font-size:12.5px;color:var(--ink);margin-top:9px}}
.msub{{font-size:11.5px;color:var(--faint);margin-top:5px;line-height:1.4}}
.section{{margin-top:46px}}
.section h2{{font-family:"Fraunces",serif;font-weight:600;font-size:23px;margin:0 0 4px}}
.section .desc{{color:var(--muted);font-size:13px;margin-bottom:18px}}
.fbar{{display:grid;grid-template-columns:160px 1fr 40px;gap:12px;align-items:center;margin:8px 0;font-family:"JetBrains Mono",monospace;font-size:12.5px}}
.flabel{{color:var(--ink)}}.fn{{text-align:right;color:var(--amber)}}
.ftrack{{height:9px;background:var(--bg);border-radius:5px;overflow:hidden}}
.ftrack i{{display:block;height:100%;background:linear-gradient(90deg,#8a6a2c,var(--amber))}}
table{{width:100%;border-collapse:collapse;margin-top:8px}}
th{{text-align:left;font-family:"JetBrains Mono",monospace;font-size:11px;letter-spacing:.08em;text-transform:uppercase;color:var(--faint);padding:8px 12px;border-bottom:1px solid var(--line2)}}
td{{padding:13px 12px;border-bottom:1px solid var(--line);vertical-align:top;font-size:13.5px}}
td.q{{width:42%}}
.rk{{font-family:"JetBrains Mono",monospace;font-size:11px;padding:2px 8px;border-radius:5px;margin-left:8px}}
.rk.hit{{color:var(--teal);background:rgba(95,184,168,.13)}}
.rk.miss{{color:var(--flag);background:rgba(223,106,74,.13)}}
.qsrc{{color:var(--faint);font-size:11.5px;margin-top:7px;font-family:"JetBrains Mono",monospace}}
.hit{{display:grid;grid-template-columns:auto auto 1fr;gap:9px;align-items:baseline;padding:3px 0;font-size:12.5px;color:var(--muted)}}
.hit.src{{color:var(--amber)}}.hit.rel{{color:var(--ink)}}
.hm{{width:14px}}.hsim{{font-family:"JetBrains Mono",monospace;font-size:11px;color:var(--faint);min-width:38px}}
.htx{{overflow:hidden;text-overflow:ellipsis}}
.pp{{font-family:"JetBrains Mono",monospace;font-size:11px;color:var(--teal);margin-top:7px}}
.fcat{{font-family:"JetBrains Mono",monospace;font-size:12px;color:var(--rose);white-space:nowrap}}
.freason{{color:var(--muted);font-size:12.5px}}
.note{{background:var(--surface);border:1px solid var(--line);border-left:3px solid var(--teal);border-radius:10px;padding:14px 18px;font-size:13px;color:var(--muted);margin-top:24px;line-height:1.6}}
.note b{{color:var(--ink)}}
.foot{{margin-top:50px;font-family:"JetBrains Mono",monospace;font-size:11px;color:var(--faint)}}
</style></head><body><div class="wrap">
<div class="head">
  <div class="eyebrow">SEARCH QUALITY EVALUATION</div>
  <h1>장면 검색 품질<em>.</em></h1>
  <div class="sub">ChromaDB 벡터 검색 · 임베딩 <code>text-embedding-3-small</code> · known-item 자동 평가.
  코퍼스 장면에서 LLM이 검색어를 생성하고(정답=출처 장면), 각 쿼리로 벡터 검색을 실행해 지표를 계산했습니다.</div>
</div>
<div class="metrics">{cards}</div>

<div class="section">
  <h2>실패 패턴 분석</h2>
  <div class="desc">정답 장면이 top-5에 포함되지 않은 쿼리들의 원인 분류 (LLM 분석)</div>
  {cat_bars or '<div class="desc">실패 케이스 없음 🎉</div>'}
</div>

<div class="section">
  <h2>실패 케이스 상세</h2>
  <table><thead><tr><th>원인</th><th>검색어</th><th>설명</th></tr></thead>
  <tbody>{fa_rows or '<tr><td colspan=3 class="freason">없음</td></tr>'}</tbody></table>
</div>

<div class="section">
  <h2>쿼리별 결과</h2>
  <div class="desc">🎯 정답 장면 · ✓ 관련(LLM 판정) · 회색 무관. 각 행의 유사도와 P@5 표시.</div>
  <table><thead><tr><th>검색어 / 정답</th><th>top-5 결과</th></tr></thead>
  <tbody>{qrows}</tbody></table>
</div>

<div class="note">
  <b>지표 해설.</b> <b>Recall@5</b>·<b>MRR</b>은 known-item 기준 — 각 쿼리의 출처 장면이 정답이라
  사람 라벨 없이 객관적으로 계산됩니다. 단, 의미가 비슷한 중복 장면이 대신 검색되면 '실패'로 잡혀
  값이 보수적으로(낮게) 나올 수 있습니다(→ 실패 패턴의 '중복_유사장면'). <b>Precision@5</b>는
  LLM-as-judge가 top-5 각 결과의 관련성을 판정한 값으로, 중복 장면도 관련으로 인정하므로 실제
  체감 품질에 더 가깝습니다. 세 지표를 함께 보세요.
</div>
<div class="foot">생성: eval/run_eval.py · 모델 {MODEL} · 자동 파이프라인</div>
</div></body></html>"""
    (OUT / "report.html").write_text(doc, encoding="utf-8")


if __name__ == "__main__":
    evaluate()
