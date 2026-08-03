from core.agent_loop import AgentLoop
from schemas.tools import TOOLS_SCHEMA
import asyncio

if __name__=="__main__":
    agent = AgentLoop()
    query = "查一下李娜的财务excel信息，文件名是财务.xml，表名是2022年"
    answer = asyncio.run(agent.run(
                session_id = 1,
                query = query,
                tools = TOOLS_SCHEMA
            ))
    print(answer)