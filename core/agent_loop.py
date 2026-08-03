import asyncio
import structlog
import json
from pathlib import Path
import sys
# root_path = Path(__file__).parent.parent
# if str(root_path) not in sys.path:
#     sys.path.append(str(root_path))

from core.llm_client import LLMClient, get_llm_client
from tools.base_tools import BaseTools, _tools
from schemas.tools import TOOLS_SCHEMA
logger = structlog.get_logger(__name__)

SYSTEM_PROMPT = """
你是一个办公助手智能体，具有调用计算器，读取excel文档进行数据查询和联网搜索的内容。
所有回答用中文进行回复，回复必须简洁。
当遇到不知道的问题就回复不知道，禁止胡编乱造。
回答问题进行逐步思考最后给出答案。
回复格式先输出[THINK]标签之后衔接思考过程。[ANSWER]标签后面衔接回答结果。
回复示例：[THINK]\n用户问xx，我需要...\n[ANSWER]\n根据数据...
回复禁止携带xml标签
"""

class AgentLoop():
    def __init__(self):
        self.llm = get_llm_client()
        
    async def run(
        self,
        session_id,
        query,
        history:list|None = None,
        tools:list|None = None,
        memory:list|None = None,
    ):
        messages = history.copy() if history else []
        messages.insert(0, {"role":"system","content":SYSTEM_PROMPT})
        messages.append({"role":"user","content":query})
        
        for loop in range(15):
            try:
                resp = await self.llm.chat(
                    messages = messages,
                    tools = tools if tools else None
                    )
            except Exception as e:
                return
            
            
            choice = resp.choices[0]
            
            chat_content = choice.message.content or ""
            tool_calls = choice.message.tool_calls
            
            if tool_calls:

                messages.append({
                    "role":"assistant",
                    "content":chat_content,
                    "tool_calls":[
                        {"id":tc.id,"type":"function","function":{"name":tc.function.name,"arguments":tc.function.arguments}}
                        for tc in tool_calls
                    ]
                })
                
                
                for tc in tool_calls:
                    tc_name = tc.function.name
                    tc_args = json.loads(tc.function.arguments)
                    
                    result = _tools.tools_excutor(tc_name, tc_args)
                    
                    messages.append({"role":"tool","tool_call_id":tc.id,"name":tc.function.name,"content":result})
                    
                    logger.debug(
                        f"loop_{loop+1}_has_toolcall",
                        tcName = tc.function.name,
                        arguments = tc_args,
                        result = result
                        
                    )
                    

                
            else:
                logger.debug(
                                        f"loop_{loop+1}_has_no_toolcall",
                                        content = choice.message.content
                                    )
                # # 兜底逻辑：检测模型嘴上说调用工具，但没有输出tool_call
                # trigger_words = ["计算器", "调用工具", "使用工具", "excel", "联网搜索"]
                # need_retry = any(word in chat_content for word in trigger_words)
                # if need_retry:
                #     logger.info("模型仅文字声明调用工具，无tool_call，继续引导")
                #     messages.append({"role":"assistant", "content": chat_content})
                #     continue
                # 正常输出最终答案
                return choice.message.content
            
        return "处理步骤达到上限，请简化问题重试"
            
            
async def main():
    agent = AgentLoop()
    query = "计算一下（34+66）*18"
    answer = await agent.run(
            session_id = 1,
            query = query,
            tools = TOOLS_SCHEMA
        )
    print(answer)

if __name__ == "__main__":
    asyncio.run(main())
    
