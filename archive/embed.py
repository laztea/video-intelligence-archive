# archive/embed.py
from archive import config

MODEL = "text-embedding-3-small"

def embed_texts(texts: list[str]) -> list[list[float]]:
    if not texts:
        return []
    client = config.get_openai()
    resp = client.embeddings.create(model=MODEL, input=texts)
    return [item.embedding for item in resp.data]

def embed_one(text: str) -> list[float]:
    return embed_texts([text])[0]
