# archive/store/vector_store.py
class VectorStore:
    def __init__(self, client, collection_name="chunks"):
        self.col = client.get_or_create_collection(name=collection_name)

    def add(self, chunk_id, embedding, text, metadata):
        self.col.upsert(
            ids=[str(chunk_id)], embeddings=[embedding],
            documents=[text], metadatas=[metadata])

    def query(self, embedding, k=20, where=None):
        res = self.col.query(query_embeddings=[embedding], n_results=k, where=where)
        out = []
        ids = res["ids"][0]
        dists = res["distances"][0]
        docs = res["documents"][0]
        metas = res["metadatas"][0]
        for i in range(len(ids)):
            out.append({
                "chunk_id": int(ids[i]), "distance": dists[i],
                "text": docs[i], "metadata": metas[i],
            })
        return out
