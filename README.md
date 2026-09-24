# Nepal Data Assistant — AI Agent

An AI agent built with Groq (GPT-OSS) and Streamlit that can search the web and do accurate math.

## What it does

- **Multi-tool agent** — decides when to use tools based on the question
- **Web search** via DuckDuckGo — fetches current info with sources
- **Calculator** — accurate math (and F→C temperature conversions!)
- **Conversation memory** — remembers context across messages
- **Tool chaining** — combines tools to answer complex questions

## Try it

- "What's the current weather in Kathmandu?"
- "What is 847 * 213 + 1200?"
- "What about Pokhara?" (after asking about Kathmandu)

## Tech stack

- Python 3.13
- **Groq API** — LLM inference (GPT-OSS 120B)
- **Streamlit** — chat UI
- **DuckDuckGo Search** — free web search

## How it works

The agent uses **tool calling**:
1. User sends a message
2. The LLM decides: answer directly, or call a tool?
3. If tool → agent executes it and feeds the result back
4. Loops until the LLM gives a final answer
5. Returns the answer + list of tools used

## Structure
