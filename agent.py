"""
Nepal Data Assistant — an AI agent with tools.
"""
import os
import json
from groq import Groq
from dotenv import load_dotenv

from tools.web_search import web_search
from tools.calculator import calculate

load_dotenv()
client = Groq(api_key=os.getenv("GROQ_API_KEY"))

MODEL = "openai/gpt-oss-120b"  # bigger model — better at tool loops

# ------------------------------------------------------------
# Tool definitions
# ------------------------------------------------------------
TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "web_search",
            "description": (
                "Search the web for current events, recent news, weather, "
                "or any factual information you don't already know. "
                "Pass ONE search query."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "A single search query string.",
                    }
                },
                "required": ["query"],
                "additionalProperties": False,
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "calculate",
            "description": (
                "Use this tool for ANY arithmetic, math, or numeric calculation. "
                "Do NOT do math in your head — always use this tool. "
                "Pass a math expression like '847 * 213 + 1200'."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "expression": {
                        "type": "string",
                        "description": "Math expression, e.g. '25 * 4 + 10' or '(1500 / 3) * 1.08'",
                    }
                },
                "required": ["expression"],
                "additionalProperties": False,
            },
        },
    },
]

AVAILABLE_TOOLS = {
    "web_search": lambda query: web_search(query, max_results=5),
    "calculate": calculate,
}


def run_tool(tool_name: str, tool_args: dict) -> str:
    if tool_name not in AVAILABLE_TOOLS:
        return f"Error: unknown tool '{tool_name}'"
    try:
        return AVAILABLE_TOOLS[tool_name](**tool_args)
    except Exception as e:
        return f"Error running {tool_name}: {str(e)}"


SYSTEM_PROMPT = (
    "You are a helpful AI assistant with two tools: web_search and calculate.\n\n"
    "RULES:\n"
    "1. ALWAYS use 'calculate' for any arithmetic — never compute in your head.\n"
    "2. Use 'web_search' for current events, weather, news, or facts you're unsure about.\n"
    "3. Call a tool ONCE with a clear query. Do NOT repeat the same tool call.\n"
    "4. After getting a tool result, use it to give your final answer immediately.\n"
    "5. If the tool result is unclear, just answer with what you have.\n"
    "Be concise. Cite sources when you use search."
)


def chat(user_message: str, history: list = None) -> dict:
    if history is None:
        history = []

    if not history:
        history.append({"role": "system", "content": SYSTEM_PROMPT})

    history.append({"role": "user", "content": user_message})

    tools_used = []
    seen_calls = set()  # track (tool_name, args) to prevent loops

    for iteration in range(5):
        response = client.chat.completions.create(
            model=MODEL,
            messages=history,
            tools=TOOLS,
            tool_choice="auto",
        )

        message = response.choices[0].message

        # ---- No tool call → final answer ----
        if not message.tool_calls:
            final = message.content or ""
            history.append({"role": "assistant", "content": final})
            return {"response": final, "history": history, "tools_used": tools_used}

        # ---- Process tool calls ----
        history.append({
            "role": "assistant",
            "content": message.content,
            "tool_calls": [
                {
                    "id": tc.id,
                    "type": "function",
                    "function": {
                        "name": tc.function.name,
                        "arguments": tc.function.arguments,
                    },
                }
                for tc in message.tool_calls
            ],
        })

        duplicate_found = False
        for tool_call in message.tool_calls:
            tool_name = tool_call.function.name
            tool_args = json.loads(tool_call.function.arguments)

            call_sig = (tool_name, json.dumps(tool_args, sort_keys=True))

            if call_sig in seen_calls:
                # Model repeated the same call — break the loop
                duplicate_found = True
                result = (
                    "You already called this tool with the same argument. "
                    "Please give your final answer now using the previous result."
                )
            else:
                seen_calls.add(call_sig)
                tools_used.append({"tool": tool_name, "args": tool_args})
                result = run_tool(tool_name, tool_args)

            history.append({
                "role": "tool",
                "tool_call_id": tool_call.id,
                "content": result,
            })

        if duplicate_found:
            # Force a final answer on the next loop iteration
            history.append({
                "role": "user",
                "content": "Stop calling tools. Give me your final answer now based on the results you already have.",
            })

    return {
        "response": "Agent reached maximum iterations.",
        "history": history,
        "tools_used": tools_used,
    }