import asyncio
import structlog
import json
from pathlib import Path
import sys
from tools.registry import tools_excutor, build_tool_message
from core.tarce import TraceLog, TraceStep
from core.llm_client import LLMClient, get_llm_client
logger = structlog.get_logger(__name__)

SYSTEM_PROMPT = """
你是一个办公助手智能体，具有调用计算器，读取excel文档进行数据查询，联网搜索，以及检索本地知识库(search_docs)的能力。
当用户问的问题涉及本地文档、内部资料、公司制度时，优先调用 search_docs 工具检索知识库，而不是联网搜索。
所有回答用中文进行回复，回复必须简洁。
当遇到不知道的问题就回复不知道，禁止胡编乱造。
回答问题进行逐步思考最后给出答案。
回复格式先输出[THINK]标签之后衔接思考过程。[ANSWER]标签后面衔接回答结果。
回复示例：[THINK]\n用户问xx，我需要...\n[ANSWER]\n根据数据...
回复禁止携带xml标签

你在调用工具时，**禁止输出任何思考过程、标签、[THINK]、[ANSWER]这类中间推理文本**。
只输出标准的function‑call工具调用，不要输出自然语言思考内容。
所有推理只在function‑call内部完成，不要把思考写在输出文本中。
如果需要向用户提问，直接在tool调用结束后的assistant content返回，不要混入工具参数
"""

class AgentLoop():
    def __init__(self):
        self.llm = get_llm_client()
        
    async def run(
        self,
        session_id,
        tools: list,
        redis,
        history: list,
        query: str,
    ):

        messages = history.copy() if history else []
        messages.insert(0, {"role":"system","content":SYSTEM_PROMPT})
        messages.append({"role":"user","content":query})
        await redis.append_history(session_id, {"role":"user","content":query})
        trace = TraceLog(session_id=session_id)
        
        for loop in range(15):
            try:
                resp = await self.llm.chat(
                    messages = messages,
                    tools = tools if tools else None
                )
                # 创建本轮LLM推理Step
                step_llm: TraceStep = trace.add_step(
                    step_type = "llm_infer",
                    input_messages_count = len(messages),
                    description = f"第{loop+1}轮模型推理",
                )
                
            except Exception as e:
                err_msg = f"模型调用异常，请稍后重试，错误信息：{str(e)}"
                logger.error(
                    f"loop_{loop+1}_llm_error",
                    error = err_msg
                )
                trace.add_step(
                    step_type = "error",
                    description = f"第{loop+1}轮模型调用异常，错误信息：{str(e)}"
                ).finish()
                continue
            
            choice = resp.choices[0]
            chat_content = choice.message.content or ""
            tool_calls = choice.message.tool_calls

            # 给当前LLM步骤填充完整数据并闭环
            step_llm.output_content = chat_content
            step_llm.output_tool_calls = tool_calls
            step_llm.model_name = "glm-4.5-air"
            step_llm.tokens_used = resp.usage.total_tokens
            step_llm.finish()
            
            if tool_calls:
                # 追加assistant工具调用消息
                messages.append({
                    "role":"assistant",
                    "content":chat_content,
                    "tool_calls":[
                        {"id":tc.id,"type":"function","function":{"name":tc.function.name,"arguments":tc.function.arguments}}
                        for tc in tool_calls
                    ]
                })
                
                # 遍历执行所有工具调用并埋点
                for tc in tool_calls:
                    tc_name = tc.function.name
                    tc_args = tc.function.arguments
                    result = await tools_excutor(tc_name, tc_args)
                    messages.append(build_tool_message(tc.id, result))
                    
                    logger.debug(
                        f"loop_{loop+1}_has_toolcall",
                        tcName = tc.function.name,
                        arguments = tc_args,
                        result = result
                    )
                    # 工具调用Step直接add并finish
                    trace.add_step(
                        step_type = "tool_call",
                        description = f"第{loop+1}轮工具调用，函数名：{tc.function.name}",
                        tool_name = tc.function.name,
                        tool_input = tc_args,
                        tool_output = result
                    ).finish()
            else:
                # 无工具调用，直接收尾本轮对话
                logger.debug(
                    f"loop_{loop+1}_has_no_toolcall",
                    content = choice.message.content
                )
                # 设置最终答案（内部自动赋值end_time）
                trace.set_final_answer(chat_content)
                await redis.append_history(session_id, {"role":"assistant","content":chat_content})
                await redis.save_trace(trace)
                return chat_content, trace
        
        # 循环15轮耗尽上限兜底
        over_limit_msg = "处理步骤达到上限，请简化问题重试"
        trace.set_error(over_limit_msg)
        await redis.save_trace(trace)
        return over_limit_msg, trace