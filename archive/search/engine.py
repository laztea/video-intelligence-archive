# archive/search/engine.py
import re
from archive import config
from archive.search.explain import build_explanation

_TOK = re.compile(r"[0-9A-Za-z가-힣]+")

def _minmax(scores: dict) -> dict:
    if not scores:
        return {}
    vals = list(scores.values())
    lo, hi = min(vals), max(vals)
    if hi == lo:
        return {k: 1.0 for k in scores}
    return {k: (v - lo) / (hi - lo) for k, v in scores.items()}

class SearchEngine:
    def __init__(self, sqlite_store, vector_store, embed_fn):
        self.sql = sqlite_store
        self.vec = vector_store
        self.embed_fn = embed_fn

    def search(self, query, mode="hybrid", filters=None, limit=20, keyword_query=None):
        """query: 의미(벡터) 검색용 문장.  keyword_query: FTS 키워드 검색용 어휘
        (없으면 query 사용). 분리해 두면 hybrid에서 벡터는 문장, 키워드는 추출
        어휘를 쓸 수 있다."""
        filters = filters or {}
        if mode == "filter":
            return self._filter(filters, limit)
        kwq = keyword_query if keyword_query is not None else query
        kw = self._keyword_scores(kwq, limit) if mode in ("keyword", "hybrid") else {}
        vec = self._vector_scores(query, limit, filters) if mode in ("vector", "hybrid") else {}
        return self._assemble(kw, vec, mode, limit)

    def _keyword_scores(self, query, limit):
        if not query:
            return {}
        terms = list(dict.fromkeys(_TOK.findall(query)))  # 중복 제거, 순서 유지
        # OR 검색으로 후보를 넉넉히 뽑은 뒤 coverage(매칭 키워드 수)로 재랭킹
        hits = self.sql.fts_search(query, limit=max(limit * 3, 30))
        out = {}
        for h in hits:
            text = (h["text"] or "").lower()
            matched = [t for t in terms if t.lower() in text]
            coverage = len(matched)
            # 매칭 키워드 수 우선, 동률이면 bm25(낮을수록 좋음 → 빼서 가산)
            raw = coverage * 100.0 - (h["bm25"] or 0.0)
            out[h["id"]] = {"bm25": h["bm25"], "raw": raw,
                            "matched": matched, "coverage": coverage}
        return out

    def _vector_scores(self, query, limit, filters):
        if not query:
            return {}
        emb = self.embed_fn(query)
        where = {"kind": filters["kind"]} if filters.get("kind") else None
        hits = self.vec.query(emb, k=limit, where=where)
        out = {}
        for h in hits:
            sim = 1.0 / (1.0 + h["distance"])
            out[h["chunk_id"]] = {"sim": sim, "raw": sim}
        return out

    def _assemble(self, kw, vec, mode, limit):
        alpha = config.hybrid_alpha()
        kw_norm = _minmax({k: v["raw"] for k, v in kw.items()})
        vec_norm = _minmax({k: v["raw"] for k, v in vec.items()})
        ids = set(kw) | set(vec)
        results = []
        for cid in ids:
            kn = kw_norm.get(cid, 0.0)
            vn = vec_norm.get(cid, 0.0)
            if mode == "keyword":
                score = kn
            elif mode == "vector":
                score = vn
            else:
                score = alpha * vn + (1 - alpha) * kn
            row = self.sql.get_chunk(cid)
            if not row:
                continue
            expl = build_explanation(
                matched_terms=kw[cid]["matched"] if cid in kw else [],
                bm25=kw[cid]["bm25"] if cid in kw else None,
                similarity=vec[cid]["sim"] if cid in vec else None)
            results.append(self._format(row, score, expl))
        results.sort(key=lambda r: r["score"], reverse=True)
        return results[:limit]

    def _filter(self, filters, limit):
        clauses, params = [], []
        if filters.get("kind"):
            clauses.append("kind=?"); params.append(filters["kind"])
        if filters.get("video_id"):
            clauses.append("video_id=?"); params.append(filters["video_id"])
        if filters.get("flagged"):
            clauses.append("id IN (SELECT chunk_id FROM chunk_flags)")
        where = (" WHERE " + " AND ".join(clauses)) if clauses else ""
        rows = self.sql.conn.execute(
            f"SELECT * FROM chunks{where} ORDER BY video_id, seq LIMIT ?",
            (*params, limit)).fetchall()
        return [self._format(dict(r), score=None,
                expl={"source": "filter", "text": "필터 조건 일치"}) for r in rows]

    def _format(self, row, score, expl):
        flags = self.sql.flags_for_chunk(row["id"])
        return {
            "chunk_id": row["id"], "video_id": row["video_id"], "kind": row["kind"],
            "start_s": row["start_s"], "end_s": row["end_s"], "text": row["text"],
            "flags": flags, "score": score, "source": expl["source"],
            "explanation": expl,
        }
