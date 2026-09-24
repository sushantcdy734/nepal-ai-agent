"""
Calculator tool — does accurate math for the agent.
LLMs often make arithmetic mistakes; this fixes that.
"""


def calculate(expression: str) -> str:
    """
    Safely evaluate a mathematical expression.
    Only allows numbers and basic operators (no code execution).
    """
    # Whitelist: only digits, operators, parentheses, decimal point, spaces
    allowed = set("0123456789+-*/(). eE")
    if not all(c in allowed for c in expression):
        return "Error: only numbers and + - * / ( ) are allowed"

    try:
        result = eval(expression, {"__builtins__": {}}, {})
        return f"{expression} = {result}"
    except Exception as e:
        return f"Error evaluating '{expression}': {str(e)}"