import math

def calculator(expression: str) -> str:
    try:
        safe_dict = {"abs":abs, "round":round, "max":max, "min":min,
                     "pow":pow, "sum":sum, "sqrt":math.sqrt,
                     "sin":math.sin, "cos":math.cos, "pi":math.pi, "e":math.e}
        return str(eval(expression, {"__builtins__": {}}, safe_dict))
    
    except Exception as e:
        return f"计算错误：{e}"