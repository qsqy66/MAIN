def web_search(query: str) -> str:
    MOCK = {
        "python": "Python由Guido van Rossum于1991年创建。",
        "langchain": "LangChain是LLM应用框架，支持Prompt/Chain/Agent。",
        "rag": "RAG结合检索与生成，提升回答准确性。",
        "glm": "GLM是智谱AI开发的大语言模型。",
        "竞品": "竞品A发布新版本，支持多模态；竞品B降价30%。",
    }
    for key, val in MOCK.items():
        if key in query.lower():
            return f"搜索'{query}': {val}"
    return f"搜索'{query}': 暂无结果。"