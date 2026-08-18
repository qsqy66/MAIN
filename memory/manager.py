"""
记忆策略层：短期记忆(会话摘要) + 长期记忆(用户画像 / 情景记忆) 的读、写编排。

职责边界：
- 本模块只做「策略」：什么时候提取、怎么去重截断、按什么召回注入。
- 实际的 Redis 存取全部委托给 memory/redis.py（纯 DAO），本模块不直接写 key。
- 向量化复用 core.llm_client.embed_texts，相似度复用 core.rag.cosine_similarity。

设计约束：load / update 全程 try/except，任何记忆操作失败都只记日志，
绝不影响主对话流程与最终答案（MEMORY_ENABLED=False 时行为与无记忆完全一致）。
"""
import json
import time

import structlog

from config import settings
from core.llm_client import get_llm_client
from core.rag import cosine_similarity

logger = structlog.get_logger(__name__)


def _parse_extraction(content: str) -> dict:
    """防御性解析模型输出的 JSON：剥代码围栏、取首个 {...}，失败返回空结构。"""
    try:
        text = content.strip()
        if text.startswith("```"):
            text = text.strip("`").lstrip()
            if text.startswith("json"):
                text = text[4:].lstrip()
        start, end = text.find("{"), text.rfind("}")
        if start == -1 or end == -1:
            return {}
        data = json.loads(text[start:end + 1])
        return data if isinstance(data, dict) else {}
    except Exception as e:
        logger.error("memory_extract_parse_error", error=str(e))
        return {}


class MemoryManager:
    def __init__(self):
        self.llm = get_llm_client()

    # ==================== 读取：加载记忆并组织注入内容 ====================
    async def load(self, user_id: str, session_id: str, query: str, redis) -> dict:
        """返回 {profile, summary, episodic_hits}，均为空时上层不注入任何内容。"""
        if not settings.MEMORY_ENABLED:
            return {}
        try:
            profile = await redis.get_profile(user_id)
            summary = await redis.get_session_summary(session_id)
            hits = []
            records = await redis.get_episodic(user_id)
            if records:
                # 情景记忆按「与当前问题的相关性」召回 top-k
                query_vec = (await self.llm.embed_texts([query]))[0]
                scored = [
                    (cosine_similarity(query_vec, r.get("embedding") or []), r)
                    for r in records
                ]
                scored.sort(key=lambda x: x[0], reverse=True)
                hits = [
                    {"text": r["text"]}
                    for s, r in scored[:settings.MEMORY_TOP_K]
                    if s >= settings.RAG_RELEVANCE_THRESHOLD
                ]
            return {"profile": profile, "summary": summary, "episodic_hits": hits}
        except Exception as e:
            logger.error("memory_load_error", error=str(e))
            return {}

    # ==================== 写入：回答完成后提取并持久化 ====================
    async def update(
        self,
        user_id: str,
        session_id: str,
        query: str,
        answer: str,
        full_history: list,
        redis,
    ) -> dict:
        """从本轮对话提取记忆并入库；返回计数供 TRACE 记录。"""
        result = {"profile_added": 0, "episodic_added": 0, "summary_updated": False}
        if not settings.MEMORY_ENABLED:
            return result
        try:
            if settings.MEMORY_EXTRACT:
                data = await self._extract(query, answer)
                await self._update_profile(redis, user_id, data.get("profile_updates"), result)
                await self._update_episodic(redis, user_id, session_id, data.get("episodic"), result)
            result["summary_updated"] = await self._maybe_summarize(redis, session_id, full_history)
        except Exception as e:
            logger.error("memory_update_error", error=str(e))
        return result

    async def _extract(self, query: str, answer: str) -> dict:
        """用轻量模型把「本轮问答」转成结构化记忆候选。"""
        sys_prompt = (
            "你是记忆抽取器。从一段「用户提问 + 助手回答」中，抽取值得长期记住的信息。\n"
            "只输出一个 JSON 对象，不要输出任何其它文字、代码围栏或解释。格式：\n"
            '{"profile_updates":[{"key":"标签","value":"事实"}],"episodic":["一句话事实"]}\n'
            "规则：\n"
            "- profile_updates：用户的稳定属性（姓名、部门、职位、偏好、习惯默认值等），key 用 2~5 字标签。\n"
            "- episodic：一次性但未来可能有用的事实、承诺、任务、决定，每条一句完整的话。\n"
            "- 没有可提取的信息就返回空数组。禁止编造。"
        )
        try:
            content = await self.llm.lite_chat(
                sys_prompt,
                f"用户提问：{query}\n助手回答：{answer}",
            )
        except Exception as e:
            logger.error("memory_extract_llm_error", error=str(e))
            return {}
        return _parse_extraction(content or "")

    async def _update_profile(self, redis, user_id: str, updates, result: dict):
        """合并画像：按 key 覆盖、保留最新 LT_PROFILE_MAX 条。"""
        if not updates:
            return
        profile = await redis.get_profile(user_id)
        added = 0
        for u in updates:
            if not isinstance(u, dict):
                continue
            key = str(u.get("key") or "").strip()
            value = str(u.get("value") or "").strip()
            if not key or not value:
                continue
            if profile.get(key) != value:
                profile[key] = value
                added += 1
        if added:
            trimmed = dict(list(profile.items())[-settings.LT_PROFILE_MAX:])
            await redis.save_profile(user_id, trimmed)
            result["profile_added"] = added

    async def _update_episodic(self, redis, user_id: str, session_id: str, texts, result: dict):
        """追加情景记忆：与已有记录余弦>0.9 判为重复跳过；保留最新 LT_EPISODIC_MAX 条。"""
        texts = [t.strip() for t in (texts or []) if isinstance(t, str) and t.strip()]
        if not texts:
            return
        records = await redis.get_episodic(user_id)
        vectors = await self.llm.embed_texts(texts)
        added = 0
        for text, vec in zip(texts, vectors):
            vec = [float(x) for x in vec]
            if any(cosine_similarity(vec, r.get("embedding") or []) > 0.9 for r in records):
                continue
            records.append({
                "text": text,
                "embedding": vec,
                "ts": time.time(),
                "session_id": session_id,
            })
            added += 1
        if added:
            trimmed = records[-settings.LT_EPISODIC_MAX:]
            await redis.save_episodic(user_id, trimmed)
            result["episodic_added"] = added

    async def _maybe_summarize(self, redis, session_id: str, full_history: list) -> bool:
        """会话轮数达到阈值后，把全部历史压缩为一段摘要（短期记忆）。"""
        msgs = [
            m for m in full_history
            if m.get("role") in ("user", "assistant") and (m.get("content") or "").strip()
        ]
        if len(msgs) < settings.SUMMARY_TRIGGER_ROUNDS * 2:
            return False
        lines = []
        for m in msgs:
            label = "用户" if m["role"] == "user" else "助手"
            lines.append(f"{label}：{m['content'].strip()}")
        summary = await self.llm.lite_chat(
            "把下面的对话压缩成一段中文摘要，保留关键信息（主题、结论、用户偏好与决定），不超过150字。只输出摘要正文。",
            "\n".join(lines),
        )
        if summary and summary.strip():
            await redis.save_session_summary(session_id, summary.strip())
            return True
        return False


_memory_manager: MemoryManager | None = None


def get_memory_manager() -> MemoryManager:
    """全局单例，复用同一个 LLM client。"""
    global _memory_manager
    if _memory_manager is None:
        _memory_manager = MemoryManager()
    return _memory_manager
