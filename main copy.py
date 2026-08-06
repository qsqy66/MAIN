from core.agent_loop import AgentLoop
from tools.registry import TOOLS_SCHEMA
import asyncio
from memory.redis import RedisMemory
from structlog import get_logger
logger = get_logger(__name__)

async def main():
    agent = AgentLoop()
    session_id = "text_2"

    while(True):
        query = input("请输入问题：")
        if query == "exit" or query == "quit" or query == "不聊了":
            break
        redis_memory = RedisMemory()
        await redis_memory.connect()
        history = await redis_memory.get_history(session_id)
        logger.debug(f"history_loaded", count=f"读取到{len(history)}条历史记录")
        
        answer = await agent.run(
                    session_id = session_id,
                    query = query,
                    tools = TOOLS_SCHEMA,
                    memory = redis_memory,
                    history = history
                )
        print(answer)
        await redis_memory.close()

if __name__=="__main__":
    asyncio.run(main())