import json
import structlog
import time
from redis import asyncio as aioredis
from config import settings
from core.tracing import get_tracer, set_span_ok, set_span_error

logger = structlog.get_logger(__name__)
tracer = get_tracer("office_agent.redis")
class RedisMemory:
    def __init__(self, redis_url: str | None = settings.REDIS_URL):
        self.redis_url = redis_url
        self.redis: aioredis.Redis | None = None
        
    async def connect(self):
        if self.redis is None:
            self.redis = aioredis.from_url(self.redis_url)
        
    @staticmethod
    def _trace_key(trace_id: str,) -> str:
        return f"trace:{trace_id}"
    
    @staticmethod
    def _history_key(trace_id: str) -> str:
        return f"history:{trace_id}"
    
    async def append_history(self, session_id: str, message: dict):
        with tracer.start_as_current_span(
            "redis.append_history",
            attributes={"session.id": session_id, "message.role": message.get("role", "")},
        ) as span:
            t0 = time.time()
            key = self._history_key(session_id)
            value = json.dumps(message, ensure_ascii=False)
            await self.redis.rpush(key, value)
            await self.redis.expire(key, 3600)
            await self.redis.ltrim(key, -20, -1)
            elapsed_ms = (time.time() - t0) * 1000
            span.set_attribute("redis.latency_ms", round(elapsed_ms, 1))
            set_span_ok(span)
        
    async def get_history(self, session_id: str, limit: int | None = None) -> list:
        with tracer.start_as_current_span(
            "redis.get_history",
            attributes={"session.id": session_id},
        ) as span:
            t0 = time.time()
            key = self._history_key(session_id)
            if limit is not None:
                history = await self.redis.lrange(key, -limit, -1)
            else:
                history = await self.redis.lrange(key, 0, -1)

            messages = []
            for msg in history:
                try:
                    text = msg.decode("utf-8")
                    messages.append(json.loads(text))
                except json.JSONDecodeError as e:
                    logger.error(f"{session_id}_history_decode_error", error=str(e))
            elapsed_ms = (time.time() - t0) * 1000
            span.set_attributes({
                "redis.latency_ms": round(elapsed_ms, 1),
                "redis.message_count": len(messages),
            })
            set_span_ok(span)
            return messages
    
    async def close(self):
        await self.redis.close()
        
