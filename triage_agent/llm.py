"""Pick the chat model from config: local Ollama (default) or Azure OpenAI."""

from __future__ import annotations

import os

from langchain_core.language_models import BaseChatModel


def get_llm(cfg: dict) -> BaseChatModel:
    llm_cfg = cfg["llm"]
    if llm_cfg["provider"] == "azure":
        # Imported here so the Ollama path does not need Azure installed or configured.
        from langchain_openai import AzureChatOpenAI

        # Endpoint, key and API version are read from the environment (.env).
        return AzureChatOpenAI(azure_deployment=os.environ["AZURE_OPENAI_DEPLOYMENT"],
                               temperature=llm_cfg["temperature"])
    from langchain_ollama import ChatOllama

    return ChatOllama(model=llm_cfg["ollama_model"], temperature=llm_cfg["temperature"])
