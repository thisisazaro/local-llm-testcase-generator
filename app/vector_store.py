from __future__ import annotations

from typing import Dict, List

import chromadb
from chromadb.config import Settings

from app.config import SETTINGS
from app.embeddings import get_embedder

client = chromadb.PersistentClient(
    path=str(SETTINGS.chroma_dir),
    settings=Settings(anonymized_telemetry=False),
)
embedder = get_embedder()
EMBEDDING_DIM = len(embedder.embed_query("dimension probe"))
COLLECTION_NAME = f"{SETTINGS.chroma_collection}_dim_{EMBEDDING_DIM}"
collection = client.get_or_create_collection(COLLECTION_NAME)



def add_chunks(file_id: str, chunks: List[Dict]) -> None:
    if not chunks:
        return

    ids = [f"{file_id}_{idx}" for idx in range(len(chunks))]
    docs = [chunk["text"] for chunk in chunks]
    embeddings = embedder.embed_documents(docs)
    metadatas = [
        {
            "file_id": file_id,
            "page": chunk["page"],
            "source_ref": chunk["source_ref"],
        }
        for chunk in chunks
    ]

    collection.upsert(
        ids=ids,
        embeddings=embeddings,
        documents=docs,
        metadatas=metadatas,
    )



def count_chunks(file_id: str) -> int:
    result = collection.get(where={"file_id": file_id}, include=[])
    return len(result.get("ids", []))



def search(file_id: str, query: str, top_k: int = 8) -> List[Dict]:
    if top_k < 1:
        return []

    result = collection.query(
        query_embeddings=[embedder.embed_query(query or "требования")],
        n_results=top_k,
        where={"file_id": file_id},
    )

    docs = result.get("documents", [[]])[0]
    metas = result.get("metadatas", [[]])[0]
    distances = result.get("distances", [[]])[0]

    output = []
    for doc, meta, distance in zip(docs, metas, distances):
        output.append(
            {
                "text": doc,
                "source_ref": meta.get("source_ref"),
                "page": meta.get("page"),
                "distance": distance,
            }
        )
    return output
