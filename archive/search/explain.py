# archive/search/explain.py
def build_explanation(matched_terms, bm25, similarity) -> dict:
    has_kw = bm25 is not None
    has_vec = similarity is not None
    if has_kw and has_vec:
        source = "both"
    elif has_kw:
        source = "keyword"
    else:
        source = "vector"

    parts = []
    if has_kw:
        terms = ", ".join(f"'{t}'" for t in matched_terms) if matched_terms else "키워드"
        parts.append(f"{terms} 일치 (FTS {bm25:.1f})")
    if has_vec:
        parts.append(f"의미 유사도 {similarity:.2f}")
    text = " + ".join(parts) + f" → {source}"
    return {"source": source, "matched_terms": matched_terms,
            "bm25": bm25, "similarity": similarity, "text": text}
