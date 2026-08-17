from core.rag import search_docs as _rag_search_docs
async def search_docs(query: str, top_k: int | None = None) -> str:
    return await _rag_search_docs(query, top_k)
