import streamlit as st
import os
import sys
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage

# Add project root to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

# Internal imports
from models.llm import get_chat_model
from utils.rag_utils import query_faiss
from utils.web_search import web_search


# ----------------------------------------------------------------
# 🧠 CHAT RESPONSE FUNCTION
# ----------------------------------------------------------------
def get_chat_response(chat_model, messages, persona, mode, rag_enabled, web_enabled):
    """Generate a chatbot response based on persona, RAG, and web search."""
    try:
        user_query = messages[-1]["content"]
        context_text = ""

        # Retrieve context from FAISS
        if rag_enabled:
            try:
                docs = query_faiss(user_query)
                if docs:
                    context_text = "\n\n".join([f"[{d['document']}] {d['content']}" for d in docs])
                else:
                    context_text = "No relevant context found in the local knowledge base."
            except Exception as e:
                context_text = f"⚠️ RAG retrieval failed: {e}"

        # If no context and web search is enabled → fallback to web
        if web_enabled and ("No relevant" in context_text or "RAG" in context_text):
            context_text = web_search(user_query)

        # Persona-based system prompts
        persona_prompts = {
            "Resume Expert": (
                "You are a Resume Expert AI mentor. Help users craft strong, keyword-rich resumes. "
                "Provide structured, actionable feedback. Use professional tone and bullet points."
            ),
            "Interview Coach": (
                "You are an Interview Coach AI mentor. Help users prepare for job interviews, "
                "ask mock questions, give sample STAR answers, and boost confidence."
            ),
            "Career Counselor": (
                "You are a Career Counselor AI mentor. Provide career development advice, "
                "guidance on skill-building, and insights into career paths and learning strategies."
            ),
        }

        system_prompt = persona_prompts.get(persona, "You are an intelligent career assistant.")

        # Mode instruction
        mode_instruction = (
            "Keep your response short, structured, and to the point."
            if mode == "Concise"
            else "Provide a detailed, example-driven explanation with practical advice."
        )

        # Final combined prompt
        final_prompt = (
            f"{system_prompt}\n\n"
            f"Mode: {mode_instruction}\n\n"
            f"Context:\n{context_text}\n\n"
            f"User Question:\n{user_query}"
        )

        # Build message history (last 10 messages only)
        formatted_messages = [SystemMessage(content=final_prompt)]
        for msg in messages[-10:]:
            if msg["role"] == "user":
                formatted_messages.append(HumanMessage(content=msg["content"]))
            else:
                formatted_messages.append(AIMessage(content=msg["content"]))

        # Get response from model
        response = chat_model.invoke(formatted_messages)
        return response.content

    except Exception as e:
        return f"⚠️ Error generating response: {str(e)}"


# ----------------------------------------------------------------
# 📘 INSTRUCTIONS PAGE
# ----------------------------------------------------------------
def instructions_page():
    st.title("📘 NeoStats AI Career Mentor – Instructions")
    st.markdown("""
    Welcome to your **AI Career Mentor**!  
    This chatbot helps with:
    - 🧠 Resume improvement advice
    - 🎯 Mock interview practice
    - 💼 Career counseling & skill planning

    ### ⚙️ Tech Stack
    - **Streamlit** for UI  
    - **LangChain** for LLM orchestration  
    - **Groq / OpenAI** as LLM providers  
    - **FAISS** for knowledge retrieval (RAG)  
    - **SerpAPI** for live web search  

    ### 🔑 How to Use
    1. Set your API keys in `.env` file.  
    2. Run:  
       ```bash
       streamlit run app.py
       ```
    3. Choose your **Persona** and **Mode** from the sidebar.  
    4. Type your question in the chat box and start interacting!

    ### 🧰 Available Features
    - 💬 Multi-persona behavior (Resume Expert / Interview Coach / Career Counselor)
    - 🔍 RAG-based knowledge retrieval
    - 🌐 Real-time web search fallback
    - 🧠 Memory (remembers last few messages)
    - 🪄 Concise / Detailed modes
    """)


# ----------------------------------------------------------------
# 💬 MAIN CHAT PAGE
# ----------------------------------------------------------------
def chat_page():
    st.title("🧭 AI Career Mentor Chatbot")
    st.caption("Your intelligent career guide powered by Llama 3 + RAG + Web Search.")

    # Sidebar settings
    st.sidebar.header("Chat Settings")
    persona = st.sidebar.selectbox(
        "Choose your Mentor Persona:",
        ["Resume Expert", "Interview Coach", "Career Counselor"],
        index=1
    )
    mode = st.sidebar.radio("Response Mode:", ["Concise", "Detailed"], index=1)
    rag_enabled = st.sidebar.checkbox("Enable Knowledge Base (RAG)", value=True)
    web_enabled = st.sidebar.checkbox("Enable Web Search (SerpAPI)", value=True)

    # Get chat model
    chat_model = get_chat_model()

    # Initialize chat history
    if "messages" not in st.session_state:
        st.session_state.messages = []

    # Display chat history
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # Input box
    if prompt := st.chat_input("Type your message here..."):
        st.session_state.messages.append({"role": "user", "content": prompt})
        st.session_state.messages = st.session_state.messages[-10:]  # Keep memory small

        with st.chat_message("user"):
            st.markdown(prompt)

        # Bot response
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                response = get_chat_response(
                    chat_model,
                    st.session_state.messages,
                    persona,
                    mode,
                    rag_enabled,
                    web_enabled
                )
                st.markdown(response)

        st.session_state.messages.append({"role": "assistant", "content": response})


# ----------------------------------------------------------------
# 🧩 MAIN ROUTER
# ----------------------------------------------------------------
def main():
    st.set_page_config(
        page_title="AI Career Mentor Chatbot",
        page_icon="🧭",
        layout="wide",
        initial_sidebar_state="expanded"
    )

    with st.sidebar:
        st.title("Navigation")
        page = st.radio("Go to:", ["Chat", "Instructions"], index=0)
        st.divider()
        if page == "Chat":
            if st.button("🗑️ Clear Chat History", use_container_width=True):
                st.session_state.messages = []
                st.rerun()

    if page == "Instructions":
        instructions_page()
    elif page == "Chat":
        chat_page()


# ----------------------------------------------------------------
# 🚀 ENTRY POINT
# ----------------------------------------------------------------
if __name__ == "__main__":
    main()
