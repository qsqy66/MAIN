import json
import structlog
from tools.calculator import calculator
from tools.web_search import web_search
from tools.excel_reader import read_excel
from tools.rag_tool import search_docs
import re

logger = structlog.get_logger(__name__)
tools_registry = {
        "calculator": calculator,
        "read_excel": read_excel,
        "web_search": web_search,
        "search_docs": search_docs,
}
    
async def tools_excutor(
        name:str,
        arguments:str,
    ):
        func = tools_registry.get(name)
        if func is None:
            result = f"未知工具"
            return result
        
        try:
            arguments = json.loads(arguments)
        except json.JSONDecodeError as e:
            logger.error(f"tool_arguments_decode_error", tool_name=name, error=str(e))
            return f"工具参数解析异常：{str(e)}"
        
        try:
            result = await func(**arguments)
            logger.info(f"tool_execution_success", tool_name=name, result=result)
            return result
        except Exception as e:
            logger.error(f"tool_execution_error", tool_name=name, error=str(e))
            return f"工具执行异常：{str(e)}"
        
def build_tool_message(tool_call_id: str, content: str) -> dict:
    """组装OpenAI标准tool返回消息"""
    return {
        "role": "tool",
        "tool_call_id": tool_call_id,
        "content": content
    }
    
    
    
TOOLS_SCHEMA = [
    {
        "type":"function",
        "function":{
            "name":"calculator",
            "description":"执行数学计算。支持+-*/、**、sqrt、sin等。",
            "parameters":{
                "type":"object",
                "properties":{
                    "expression":{"type":"string","description":"数学表达式"}
                },
                "required":["expression"]
            }
        }
    },
    {
        "type":"function",
        "function":{
            "name":"read_excel",
            "description":"excel文件查询器，返回表格内数据",
            "parameters":{
                "type":"object",
                "properties":{
                    "filename":{"type":"string","decription":"要查询文件的名字"},
                    "sheet_name":{"type":"string","decription":"要查询的表名，如果名称错误，工具会返回全部可用sheet"}
                },
                "required":["filename","sheet_name"]
            }
        }
    },
    {
        "type":"function",
        "function":{
            "name":"web_search",
            "decription":"联网搜索信息。用于查找竞品动态、行业资讯、外部信息。",
            "parameters":{
                "type":"object",
                "properties":{
                    "query":{"type":"string","description":"要查询的内容"}
                },
                "required":["query"]
            }
        }
    },
    {
        "type":"function",
        "function":{
            "name":"search_docs",
            "description":"检索本地知识库(docs目录)里的私有文档，回答公司制度、产品资料、业务文档等内部问题。当问题涉及本地/内部资料时优先使用，比联网搜索更准确。",
            "parameters":{
                "type":"object",
                "properties":{
                    "query":{"type":"string","description":"检索关键词或问题"},
                    "top_k":{"type":"integer","description":"返回片段数量，默认5，可不传"}
                },
                "required":["query"]
            }
        }
    }
]