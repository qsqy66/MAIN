from tools.calculator import calculator
from tools.web_search import web_search
from tools.excel_reader import read_excel

TOOLS_MAP = {
            "calculator": calculator,
            "read_excel": read_excel,
            "web_search": web_search,
        }

class BaseTools():
    def __init__(self):
        self.tools_map = TOOLS_MAP
    
    def tools_excutor(
        self,
        name:str,
        arguments:dict,
    ):
        func = self.tools_map.get(name)
        if func is None:
            result = f"未知工具"
        result = func(**arguments)
        
        return result
    
_tools = BaseTools()
        
        