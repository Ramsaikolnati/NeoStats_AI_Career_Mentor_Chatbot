"""
Configuration file for environment variables and constants
used in the NeoStats AI Career Mentor Chatbot project.
"""

import os
from dotenv import load_dotenv

# Load .env file
load_dotenv()

# OpenAI
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
OPENAI_CHAT_MODEL = os.getenv("OPENAI_CHAT_MODEL", "gpt-4o")
OPENAI_EMBED_MODEL = os.getenv("OPENAI_EMBED_MODEL", "text-embedding-3-small")
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")


# Groq
GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")
GROQ_MODEL = os.getenv("GROQ_MODEL", "llama-3.1-8b-instant")


# FAISS
FAISS_INDEX_PATH = os.getenv("FAISS_INDEX_PATH", "faiss_index.bin")

# Knowledge Base
KB_DIR = os.getenv("KB_DIR", "kb")

# SerpAPI (for live web search)
SERPAPI_API_KEY = os.getenv("SERPAPI_API_KEY", "")

# App settings
MAX_CONTEXT_CHUNKS = int(os.getenv("MAX_CONTEXT_CHUNKS", "3"))
