# 🧭 AI Career Mentor Chatbot

**A Streamlit-based conversational AI project built for NeoStats AI Engineer Use Case.**  
This chatbot acts as a **Career Mentor**, offering resume advice, interview prep, and career guidance.  
It integrates **RAG (Retrieval-Augmented Generation)** and **Live Web Search** for contextual responses.

---

## 🚀 Features
- 💬 Multi-Persona Chatbot (Resume Expert / Interview Coach / Career Counselor)
- 🔍 Retrieval-Augmented Generation (FAISS-based Knowledge Base)
- 🌐 Real-Time Web Search via SerpAPI
- 🧠 Supports OpenAI & Groq Models
- ⚙️ Concise / Detailed Response Modes

---

## 🧩 Tech Stack
- **Frontend:** Streamlit  
- **LLMs:** OpenAI GPT-4o / Groq LLaMA 3  
- **RAG Engine:** FAISS + SentenceTransformers  
- **Web Search:** SerpAPI  
- **Language:** Python  

---

## 🛠️ Local Setup
```bash
git clone https://github.com/Ramsaikolnati/NeoStats_AI_Career_Mentor_Chatbot.git
cd NeoStats_AI_Career_Mentor_Chatbot
python -m venv .venv
source .venv/bin/activate  # or .venv\Scripts\activate on Windows
pip install -r requirements.txt
