"""Streamlit UI for the Agentic RAG system."""
import os
import requests
import streamlit as st


API_URL = os.environ.get("API_URL", "http://localhost:8000/api/v1/chat")
API_KEY = os.environ.get("UI_API_KEY", "secret-admin-key")

st.set_page_config(page_title="Multimodal RAG Assistant", layout="wide")
st.title("Research Assistant")

mode = st.radio("Search Mode", ["fast", "deep"], horizontal=True)

if "messages" not in st.session_state:
    st.session_state.messages = []

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

if prompt := st.chat_input("Ask a question..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        placeholder = st.empty()
        placeholder.markdown("Thinking...")

        try:
            res = requests.post(
                API_URL,
                json={
                    "query": prompt,
                    "history": st.session_state.messages[:-1],
                    "mode": mode
                },
                headers={"X-API-Key": API_KEY}
            )
            res.raise_for_status()
            data = res.json()
            
            ans = data.get("answer", "")
            citations = data.get("citations", [])
            
            if citations:
                ans += "\n\n### Sources\n"
                for c in citations:
                    ans += f"- **[{c['index']}]** {c['source']} (Score: {c['score']:.2f})\n"

            placeholder.markdown(ans)
            st.session_state.messages.append({"role": "assistant", "content": ans})
        except Exception as e:
            placeholder.markdown(f"**Error:** {str(e)}")
