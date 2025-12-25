from langchain_groq import ChatGroq
from langchain_openai import ChatOpenAI
from langchain_anthropic import ChatAnthropic

from langchain_core.language_models.chat_models import BaseChatModel
from dotenv import load_dotenv

load_dotenv()


def get_llm(model_name: str, temperature: float = 0.7) -> BaseChatModel:
    if model_name == "gpt-4o-mini":
        return ChatOpenAI(model="gpt-4o-mini", temperature=temperature)
    # elif model_name == "llama3-8b-8192":
    #     return ChatGroq(model="llama3-8b-8192", temperature=temperature)
    elif model_name == "llama-3.1-8b-instant":
        return ChatGroq(model="lllama-3.1-8b-instant", temperature=temperature)    
    elif model_name == "claude-sonnet-4-20250514":
        return ChatAnthropic(model="claude-sonnet-4-20250514", temperature=temperature)
    else:
        raise ValueError(f"Unknown model name: {model_name}")