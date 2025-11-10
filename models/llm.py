import os
from langchain_groq import ChatGroq
from langchain_openai import ChatOpenAI
from config.config import OPENAI_API_KEY

def get_chat_model():
    """
    Initialize and return a chat model based on available API keys.
    Priority: Groq -> OpenAI.
    """
    groq_key = os.getenv("GROQ_API_KEY", "")
    openai_key = OPENAI_API_KEY

    try:
        if groq_key:
            # Use Groq model if key is set
            print("✅ Using Groq model (llama-3.1-8b-instant)")
            return ChatGroq(
                api_key=groq_key,
                model="llama-3.1-8b-instant"
            )
        elif openai_key:
            # Fallback to OpenAI if Groq key not found
            print("✅ Using OpenAI model (gpt-4o-mini)")
            return ChatOpenAI(
                api_key=openai_key,
                model="gpt-4o-mini",
                temperature=0.6
            )
        else:
            raise ValueError("No valid API keys found for Groq or OpenAI.")
    except Exception as e:
        raise RuntimeError(f"Failed to initialize chat model: {str(e)}")
