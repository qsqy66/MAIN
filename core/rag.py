import json
import structlog
from pathlib import Path

import numpy as np

from config import settings
from core.llm_client import get_llm_client

logger = structlog.get_logger(__name__)


# ======================================================================
# 第一步：文档加载 —— 把 DOCS_DIR 目录里的文件读成纯文本
# ======================================================================

# 这里只列出「无需额外安装依赖」就能读取的文件后缀。
# .pdf 需要 pypdf / PyPDF2，.docx 需要 python-docx，本环境暂未安装，
# 遇到这些格式会记一条日志并跳过（见 load_documents 末尾）。
_TEXT_EXTS = {".txt", ".md", ".markdown"}
_TABLE_EXTS = {".csv", ".xlsx", ".xls"}


def _read_text_file(path: Path) -> str:
    """读取纯文本类文件（.txt / .md / .markdown），统一按 utf-8 解码。"""
    # errors="ignore" 表示遇到无法解码的字符时跳过，而不是直接抛异常
    return path.read_text(encoding="utf-8", errors="ignore")


def _read_table_file(path: Path) -> str:
    """读取表格类文件（.csv / .xlsx / .xls），把数据转成文本。

    思路：表格本质是「有行有列的数据」，直接 to_string() 转成对齐文本，
    大模型就能像读文字一样理解表格内容了。
    """
    import pandas as pd

    if path.suffix.lower() == ".csv":
        df = pd.read_csv(path)
        return f"文件：{path.name}\n" + df.to_string()

    # Excel 可能包含多个工作表（sheet），sheet_name=None 表示一次性读取全部
    sheets = pd.read_excel(path, sheet_name=None)
    parts = []
    for sheet_name, df in sheets.items():
        parts.append(f"工作表：{sheet_name}\n{df.to_string()}")
    return f"文件：{path.name}\n" + "\n\n".join(parts)


def load_documents(docs_dir) -> list[dict]:
    """扫描文档目录，返回 [{"source": 文件名, "text": 全文}, ...] 列表。

    返回的每个元素是一篇「文档」，后续会依次被切块、向量化、入库。
    目录不存在或没有可读文件时返回空列表，由调用方负责提示用户。
    """
    docs_dir = Path(docs_dir)
    docs = []
    if not docs_dir.exists():
        return docs

    for path in sorted(docs_dir.iterdir()):
        if not path.is_file():
            continue  # 跳过子目录等非文件项

        suffix = path.suffix.lower()
        try:
            if suffix in _TEXT_EXTS:
                text = _read_text_file(path)
            elif suffix in _TABLE_EXTS:
                text = _read_table_file(path)
            else:
                # 不支持的格式（如 .pdf / .docx）需要额外解析库，这里跳过并提示
                logger.warning("rag_unsupported_file", file=path.name)
                continue
        except Exception as e:
            # 单个文件读取失败不应影响其它文件，记日志后继续
            logger.error("rag_read_file_error", file=path.name, error=str(e))
            continue

        # 去掉空白文档，避免向量化空文本造成浪费
        if text.strip():
            docs.append({"source": path.name, "text": text})

    return docs


# ======================================================================
# 第二步：文本切块 —— 把长文档切成带重叠的小片段（chunk）
# ======================================================================

def chunk_text(text: str, chunk_size: int = 500, overlap: int = 50) -> list[str]:
    """把一段长文本切成多个小片段。
    
    切分逻辑（滑窗）：
        chunk_size=500, overlap=50 时：
        第0块 取 [0, 500)，第1块 取 [450, 950)，第2块 取 [900, 1400)……
        相邻两块之间重叠 50 个字符。
    """
    text = text.strip()
    if not text:
        return []

    chunks = []
    start = 0
    n = len(text)
    while start < n:
        end = start + chunk_size
        chunks.append(text[start:end])
        # 当前这一块已经切到文末了，就不要再往后面滑了，
        # 否则会多出一个「完全被上一块覆盖」的冗余小尾巴
        if end >= n:
            break
        # 步长 = chunk_size - overlap，这样相邻块之间正好重叠 overlap 个字符
        start += chunk_size - overlap
    return chunks


# ======================================================================
# 第三步：相似度计算 —— 余弦相似度
# ======================================================================

def cosine_similarity(a, b) -> float:
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    if norm_a == 0 or norm_b == 0:
        return 0.0  # 零向量无法定义夹角，直接视为不相关
    return float(np.dot(a, b) / (norm_a * norm_b))


# ======================================================================
# 第四步：RAG 引擎 —— 把上面三步串起来，负责「索引」和「检索」
# ======================================================================

class RAGEngine:
    """RAG 引擎：负责文档索引（向量化入库）和检索（查询 top-k 相关片段）。

    内部用一个列表保存所有「片段」，并持久化到磁盘 JSON 文件，重启不丢：
        self.chunks: list[dict]，每个元素形如
            {
                "id":        唯一编号（来源文件 + 序号）,
                "source":    来源文件名,
                "text":      片段原文,
                "embedding": 该片段的向量（float 列表）,
            }
    后续想换成 chromadb / milvus 等生产级向量库时，只需替换
           retrieve() / index_documents() 内部实现，对外接口不变。
    """

    def __init__(self):
        self.chunks: list[dict] = []
        # 复用配置里的 CHROMA_PERSIST_DIR 作为向量索引的存放目录，
        # CHROMA_COLLECTION 作为索引文件名（保持配置向后兼容）
        self._persist_dir = Path(settings.CHROMA_PERSIST_DIR)
        self._persist_file = self._persist_dir / f"{settings.CHROMA_COLLECTION}.json"
        self._load()

    # -------------------- 持久化 --------------------
    def _load(self):
        """启动时从磁盘恢复已索引的向量，避免每次重启都重新向量化（费时费钱）。"""
        if self._persist_file.exists():
            try:
                self.chunks = json.loads(self._persist_file.read_text(encoding="utf-8"))
                logger.info("rag_index_loaded", count=len(self.chunks))
            except Exception as e:
                logger.error("rag_index_load_error", error=str(e))
                self.chunks = []

    def _save(self):
        """把当前向量库整体写入磁盘 JSON 文件。"""
        self._persist_dir.mkdir(parents=True, exist_ok=True)
        # ensure_ascii=False 保证中文原样存储，文件可读
        self._persist_file.write_text(
            json.dumps(self.chunks, ensure_ascii=False),
            encoding="utf-8",
        )

    # -------------------- 索引 --------------------
    async def index_documents(self, force: bool = False) -> int:
        """扫描 DOCS_DIR，把文档向量化后写入向量库，返回新索引的片段数。

        force=False 时：若库中已有数据则直接跳过（幂等，避免重复索引）；
        force=True 时：强制清空重建（文档更新后手动触发）。
        """
        if self.chunks and not force:
            return 0

        docs = load_documents(settings.DOCS_DIR)
        if not docs:
            logger.info("rag_no_documents", dir=settings.DOCS_DIR)
            return 0

        llm = get_llm_client()
        new_chunks: list[dict] = []
        idx = 0
        for doc in docs:
            pieces = chunk_text(doc["text"], settings.CHUNK_SIZE, settings.CHUNK_OVERLAP)
            if not pieces:
                continue
            # 把「整篇文档的所有片段」一次性批量向量化，减少网络往返次数
            embeddings = await llm.embed_texts(pieces)
            for text, vec in zip(pieces, embeddings):
                new_chunks.append({
                    "id": f"{doc['source']}#{idx}",
                    "source": doc["source"],
                    "text": text,
                    # 向量里可能混有 numpy 类型，转成 float 便于 JSON 序列化
                    "embedding": [float(x) for x in vec],
                })
                idx += 1

        self.chunks = new_chunks
        self._save()
        logger.info("rag_indexed", chunks=len(self.chunks))
        return len(self.chunks)

    # -------------------- 检索 --------------------
    async def retrieve(self, query: str, top_k: int | None = None) -> list[dict]:
        """给定用户问题，检索出最相关的 top_k 个片段。

        流程：问题 → 转向量 → 与库中每个片段算余弦相似度 → 取分数最高的前 top_k。
        返回的每个片段附带 similarity 分数（0~1），供上层判断「够不够相关」，
        若分数普遍很低，说明知识库里可能没有相关内容（可回退联网搜索）。
        """
        top_k = top_k or settings.RAG_TOP_K
        if not self.chunks:
            return []

        llm = get_llm_client()
        # 把「问题」也转成向量（只取第一个，因为只传了一个 query）
        query_vec = (await llm.embed_texts([query]))[0]

        results = []
        for chunk in self.chunks:
            score = cosine_similarity(query_vec, chunk["embedding"])
            # 把分数一起放进结果里，保留出处与原文
            results.append({**chunk, "similarity": round(score, 4)})

        # 按相似度从高到低排序，取前 top_k
        results.sort(key=lambda x: x["similarity"], reverse=True)
        return results[:top_k]


# ======================================================================
# 单例 + 对外检索函数
# ======================================================================

# 模块级变量在第一次 import 时初始化一次，之后全局共享同一个引擎实例，
# 避免每次调用都重新加载向量库（和 config.settings、llm_client 同理）。
_rag_engine: RAGEngine | None = None


def get_rag_engine() -> RAGEngine:
    """获取全局唯一的 RAG 引擎实例（单例）。"""
    global _rag_engine
    if _rag_engine is None:
        _rag_engine = RAGEngine()
    return _rag_engine


async def search_docs(query: str, top_k: int | None = None) -> str:
    """对外提供的检索函数（供工具层调用），把检索结果格式化成人可读文本。

    若向量库为空，会先自动索引一次文档目录（懒加载），
    这样用户无需手动执行「建库」步骤，第一次提问即可用。
    """
    engine = get_rag_engine()
    if not engine.chunks:
        # 首次使用自动建库；索引 0 篇说明 docs 目录为空或没有可读文件
        count = await engine.index_documents()
        if count == 0:
            return "知识库为空：请把文档放到 docs 目录下（支持 .txt/.md/.csv/.xlsx），然后重试。"

    results = await engine.retrieve(query, top_k)
    if not results:
        return "未在知识库中检索到相关内容。"

    lines = [f"已从本地知识库检索到 {len(results)} 条相关内容："]
    for i, r in enumerate(results, 1):
        lines.append(
            f"\n[片段{i}] 来源：{r['source']}（相关度：{r['similarity']}）\n{r['text']}"
        )
    return "\n".join(lines)
