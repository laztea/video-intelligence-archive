# archive/search/engine.py
from archive import config
from archive.search.explain import build_explanation

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

    def search(self, query, mode="hybrid", filters=None, limit=20):
        filters = filters or {}
        if mode == "filter":
            return self._filter(filters, limit)
        kw = self._keyword_scores(query, limit) if mode in ("keyword", "hybrid") else {}
        vec = self._vector_scores(query, limit, filters) if mode in ("vector", "hybrid") else {}
        return self._assemble(kw, vec, mode, limit)

    def _keyword_scores(self, query, limit):
        if not query:
            return {}
        hits = self.sql.fts_search(query, limit=limit)
        # bm25는 낮을수록 좋음 → 부호 반전해 점수화
        return {h["id"]: {"row": h, "bm25": h["bm25"], "raw": -h["bm25"]} for h in hits}

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
            matched = self._matched_terms(row["text"], kw.get(cid))
            expl = build_explanation(
                matched_terms=matched,
                bm25=kw[cid]["bm25"] if cid in kw else None,
                similarity=vec[cid]["sim"] if cid in vec else None)
            results.append(self._format(row, score, expl))
        results.sort(key=lambda r: r["score"], reverse=True)
        return results[:limit]

    def _matched_terms(self, text, kw_entry):
        if not kw_entry:
            return []
        return [w for w in set(text.split()) if w in text]  # 단순 텀 표기

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
