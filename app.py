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

    # Get agent response
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            try:
                result = chat(prompt, history=st.session_state.get("history", []))
                response_text = result["response"]
                tools_used = result["tools_used"]
                st.session_state["history"] = result["history"]
            except Exception as e:
                response_text = f"⚠️ Something went wrong: {str(e)}"
                tools_used = []

        st.markdown(response_text)

        if tools_used:
            with st.expander(f"🔧 Used {len(tools_used)} tool(s)"):
                for t in tools_used:
                    st.code(f"{t['tool']}({t['args']})", language="python")

    st.session_state.messages.append({"role": "assistant", "content": response_text})