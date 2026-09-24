import os
import streamlit as st
from dotenv import load_dotenv

from agent import chat

# ---------- Page config ----------
st.set_page_config(
    page_title="Nepal Data Assistant",
    page_icon="🤖",
    layout="wide",
)

# ---------- Sidebar ----------
with st.sidebar:
    st.title("🤖 Nepal Data Assistant")
    st.markdown("""
    An AI agent that can:
    - **Search the web** for current info
    - **Do accurate math** with a calculator tool
    - **Remember** the conversation
    """)

    st.markdown("---")
    st.markdown("### 💡 Try these")
    st.markdown("""
    - What's the current weather in Kathmandu?
    - What is 847 * 213 + 1200?
    - Latest news about Nepal?
    - Who won the last World Cup?
    """)

    st.markdown("---")
    if st.button("🗑️ Clear conversation"):
        st.session_state.messages = []
        st.rerun()

    st.caption("Built with Groq + GPT-OSS · Made by Sushant Chaudhary")

# ---------- Main area ----------
st.title("🤖 Nepal Data Assistant")
st.caption("Ask me anything — I'll search the web or do math when needed.")

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []  # list of {"role": "user"/"assistant", "content": str}

# Render past messages
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# Chat input
if prompt := st.chat_input("Ask me something..."):
    # Show user message
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Stream agent response
    with st.chat_message("assistant"):
        status_placeholder = st.empty()
        response_placeholder = st.empty()

        full_response = ""
        tools_used = []
        new_history = st.session_state.get("history", [])

        try:
            for event in chat(prompt, history=new_history):
                if event["type"] == "status":
                    status_placeholder.caption(event["data"])
                elif event["type"] == "text":
                    full_response += event["data"]
                    response_placeholder.markdown(full_response + "▌")
                elif event["type"] == "tool_used":
                    tools_used.append(event["data"])
                elif event["type"] == "done":
                    status_placeholder.empty()
                    response_placeholder.markdown(full_response)
                    new_history = event["data"]["history"]
        except Exception as e:
            response_placeholder.error(f"Something went wrong: {e}")
            full_response = "⚠️ The agent hit an error."

        if tools_used:
            with st.expander(f"🔧 Used {len(tools_used)} tool(s)"):
                for t in tools_used:
                    st.code(f"{t['tool']}({t['args']})", language="python")

    st.session_state["history"] = new_history
    st.session_state.messages.append({"role": "assistant", "content": full_response})