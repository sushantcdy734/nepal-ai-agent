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

MODEL = "openai/gpt-oss-120b"

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


# ------------------------------------------------------------
# System prompt — this shapes the agent's personality & style
# ------------------------------------------------------------
SYSTEM_PROMPT = """You are Nepal Data Assistant — a professional AI agent built by Sushant Chaudhary.

## Your identity
- You are knowledgeable, thoughtful, and precise.
- You have two tools: `web_search` and `calculate`.
- You explain your reasoning clearly, not just answer mechanically.

## Tool use rules
1. ALWAYS use `calculate` for any arithmetic. Never compute in your head.
2. Use `web_search` for current events, weather, news, prices, or facts you're unsure about.
3. Call `web_search` AT MOST TWICE per question. If the first results are unclear, use what you have and give your best answer.
4. After using a tool, immediately give your final answer using the result.
5. Do NOT keep calling tools hoping for a better result. Work with what you get.

## Formatting rules — VERY IMPORTANT
- Write all math in PLAIN TEXT. Never use LaTeX notation.
  - GOOD: 847 × 213 + 1200 = 181,611
  - BAD: \\(847 \\times 213 + 1{,}200 = 181{,}611\\)
- Use standard characters: × for multiplication, / for division.
- Do not use `\\times`, `\\[`, `\\]`, `\\(`, `\\)`, or wrap numbers in `{}`.
- Use markdown for structure (headers, bold, bullets) but keep the content plain.

## Response style
- Be **thorough but focused**. Match your answer length to the question.
  - Simple question → 1-3 sentences.
  - Complex question → use headings, bullets, and short paragraphs.
- Use **markdown formatting**: bold key terms, bullets for lists.
- For math, show your work: the expression, the result, and a one-line explanation.
- For factual answers from search, cite the source URL inline like [source](url).
- Never start with "Sure!" or "Great question!". Just answer.

## When you're unsure
- Say so explicitly: "I couldn't find a reliable source, but based on general knowledge..."
- Never invent facts. Hallucination is worse than an honest "I don't know."

## Tone
- Professional, clear, and calm — like a senior colleague explaining something.
- Warm but not casual. No emojis unless the user uses them first.

## Example of a good answer
User: "What's the current temperature in Kathmandu and what's 25% of 66?"

Your answer:
Kathmandu is currently 19°C (66°F) with partly cloudy skies [source](https://www.accuweather.com/).

25% of 66 = 16.5 (calculated as 66 × 0.25).

**Summary:** Around 19°C in Kathmandu — comfortable weather if you're heading out.
"""


# ------------------------------------------------------------
# Streaming chat — yields events as the agent thinks
# ------------------------------------------------------------
def chat(user_message: str, history: list = None):
    """
    Streaming chat. Yields events as the agent thinks and responds.

    Event types:
        {"type": "status",    "data": "Using web_search..."}
        {"type": "text",      "data": "partial text chunk"}
        {"type": "tool_used", "data": {"tool": ..., "args": ...}}
        {"type": "done",      "data": {"history": ..., "tools_used": [...]}}
    """
    if history is None:
        history = []

    if not history:
        history.append({"role": "system", "content": SYSTEM_PROMPT})

    history.append({"role": "user", "content": user_message})
    tools_used = []
    seen_calls = set()

    # After this many iterations, force the model to answer without tools
    FORCE_FINAL_AFTER = 3

    for iteration in range(6):
        force_final = iteration >= FORCE_FINAL_AFTER

        create_kwargs = {
            "model": MODEL,
            "messages": history,
            "stream": True,
        }
        if not force_final:
            create_kwargs["tools"] = TOOLS
            create_kwargs["tool_choice"] = "auto"

        # ---- Create a streaming completion ----
        stream = client.chat.completions.create(**create_kwargs)

        text_buffer = ""
        tool_calls_buffer = []

        # ---- Consume the stream chunk by chunk ----
        for chunk in stream:
            if not chunk.choices:
                continue

            delta = chunk.choices[0].delta

            # Tool call deltas
            if getattr(delta, "tool_calls", None):
                for tc in delta.tool_calls:
                    idx = tc.index if tc.index is not None else 0
                    while len(tool_calls_buffer) <= idx:
                        tool_calls_buffer.append({"id": "", "name": "", "arguments": ""})
                    if tc.id:
                        tool_calls_buffer[idx]["id"] = tc.id
                    if tc.function:
                        if tc.function.name:
                            tool_calls_buffer[idx]["name"] = tc.function.name
                        if tc.function.arguments:
                            tool_calls_buffer[idx]["arguments"] += tc.function.arguments

            # Text deltas — stream to the UI immediately
            if delta.content:
                text_buffer += delta.content
                yield {"type": "text", "data": delta.content}

        # ---- If the model made tool calls AND we're not forcing final, execute them ----
        if tool_calls_buffer and not force_final:
            history.append({
                "role": "assistant",
                "content": text_buffer or None,
                "tool_calls": [
                    {
                        "id": tc["id"],
                        "type": "function",
                        "function": {
                            "name": tc["name"],
                            "arguments": tc["arguments"],
                        },
                    }
                    for tc in tool_calls_buffer
                ],
            })

            for tc in tool_calls_buffer:
                tool_name = tc["name"]
                try:
                    tool_args = json.loads(tc["arguments"])
                except json.JSONDecodeError:
                    tool_args = {}

                call_sig = (tool_name, json.dumps(tool_args, sort_keys=True))

                if call_sig in seen_calls:
                    # Duplicate — nudge the model to finish
                    result = (
                        "You already called this tool with the same arguments. "
                        "Use the previous result to give your final answer now."
                    )
                else:
                    seen_calls.add(call_sig)
                    yield {"type": "status", "data": f"Using **{tool_name}**..."}
                    tools_used.append({"tool": tool_name, "args": tool_args})
                    yield {"type": "tool_used", "data": {"tool": tool_name, "args": tool_args}}
                    result = run_tool(tool_name, tool_args)

                history.append({
                    "role": "tool",
                    "tool_call_id": tc["id"],
                    "content": result,
                })

            # After 1 tool round, nudge the model to finish next time
            if iteration >= 1:
                history.append({
                    "role": "user",
                    "content": (
                        "You have enough information now. Give your final answer "
                        "using the tool results above. Do NOT call any more tools. "
                        "Do NOT use LaTeX — plain text math only."
                    ),
                })

            continue

        # ---- No tool calls (or forced final) — this is the answer ----
        history.append({"role": "assistant", "content": text_buffer})
        yield {"type": "done", "data": {"history": history, "tools_used": tools_used}}
        return

    # Ran out of iterations
    yield {"type": "done", "data": {"history": history, "tools_used": tools_used}}