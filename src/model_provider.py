from __future__ import annotations

from dataclasses import dataclass
import os


@dataclass
class ProviderConfig:
    """Shared provider configuration for LLM models."""

    provider: str
    model_name: str
    temperature: float = 0.0
    api_key: str | None = None
    base_url: str | None = None


def normalize_provider(value: str) -> str:
    """Map provider aliases and lowercase the name."""
    val = value.strip().lower().replace("-", "_").replace(" ", "_").rstrip("_")
    alias_map = {
        "anthorpic": "anthropic",
        "claude": "anthropic",
        "google": "gemini",
        "google_genai": "gemini",
        "gemini_ai": "gemini",
        "open_ai": "openai",
        "gpt": "openai",
        "open_router": "openrouter",
        "openrouter": "openrouter",
        "local": "ollama",
    }
    return alias_map.get(val, val)


def build_chat_model(config: ProviderConfig):
    """Instantiate the chat model for the selected provider.

    Supported providers:
    - `openai`: ChatOpenAI
    - `custom`: ChatOpenAI with custom base_url
    - `gemini`: ChatGoogleGenerativeAI
    - `anthropic`: ChatAnthropic
    - `ollama`: ChatOllama
    - `openrouter`: ChatOpenAI with OpenRouter base_url
    """
    provider = normalize_provider(config.provider)

    max_tokens = int(os.getenv("LLM_MAX_TOKENS", "500"))

    if provider == "openai":
        from langchain_openai import ChatOpenAI

        api_key = config.api_key or os.getenv("OPENAI_API_KEY")
        return ChatOpenAI(
            model=config.model_name,
            temperature=config.temperature,
            api_key=api_key,
            max_tokens=max_tokens,
        )

    if provider == "custom":
        from langchain_openai import ChatOpenAI

        api_key = config.api_key or os.getenv("CUSTOM_API_KEY") or "dummy"
        base_url = config.base_url or os.getenv("CUSTOM_BASE_URL")
        return ChatOpenAI(
            model=config.model_name,
            temperature=config.temperature,
            api_key=api_key,
            base_url=base_url,
            max_tokens=max_tokens,
        )

    if provider == "openrouter":
        from langchain_openai import ChatOpenAI

        api_key = config.api_key or os.getenv("OPENROUTER_API_KEY")
        base_url = config.base_url or "https://openrouter.ai/api/v1"
        return ChatOpenAI(
            model=config.model_name,
            temperature=config.temperature,
            api_key=api_key,
            base_url=base_url,
            max_tokens=max_tokens,
        )

    if provider == "gemini":
        try:
            import importlib
            mod = importlib.import_module("langchain_google_genai")
            ChatGoogleGenerativeAI = getattr(mod, "ChatGoogleGenerativeAI")
        except ImportError:
            raise ImportError(
                "langchain-google-genai is not installed. Please install it with: pip install langchain-google-genai"
            )
        api_key = config.api_key or os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
        return ChatGoogleGenerativeAI(
            model=config.model_name,
            temperature=config.temperature,
            google_api_key=api_key,
        )

    if provider == "anthropic":
        try:
            import importlib
            mod = importlib.import_module("langchain_anthropic")
            ChatAnthropic = getattr(mod, "ChatAnthropic")
        except ImportError:
            raise ImportError(
                "langchain-anthropic is not installed. Please install it with: pip install langchain-anthropic"
            )
        api_key = config.api_key or os.getenv("ANTHROPIC_API_KEY")
        return ChatAnthropic(
            model=config.model_name,
            temperature=config.temperature,
            api_key=api_key,
        )

    if provider == "ollama":
        import importlib
        ChatOllama = None
        try:
            mod = importlib.import_module("langchain_ollama")
            ChatOllama = getattr(mod, "ChatOllama")
        except ImportError:
            try:
                mod = importlib.import_module("langchain_community.chat_models")
                ChatOllama = getattr(mod, "ChatOllama")
            except ImportError:
                raise ImportError(
                    "langchain-ollama or langchain-community is not installed."
                )
        base_url = config.base_url or os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
        return ChatOllama(
            model=config.model_name,
            temperature=config.temperature,
            base_url=base_url,
        )

    raise ValueError(f"Unsupported provider: {config.provider}")
