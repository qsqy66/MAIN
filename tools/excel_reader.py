from pathlib import Path

def read_excel(filename: str, sheet_name: str = "Sheet1") -> str:
    filepath = Path(__file__).parent / filename
    if not filepath.exists():
        return f"错误：文件 '{filename}' 不存在。可用文件请在询问用户后重试。"
    try:
        import pandas as pd
        df = pd.read_excel(filepath, sheet_name=sheet_name)
        return f"'{filename}'({sheet_name}): {len(df)}行×{len(df.columns)}列\n" + df.head(20).to_string()
    except Exception as e:
        return f"读取错误：{e}"