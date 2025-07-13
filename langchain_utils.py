import os
from dotenv import load_dotenv
from langchain_postgres import PGVector
from langchain_openai import OpenAIEmbeddings, ChatOpenAI

load_dotenv()


def get_embeddings_client() -> OpenAIEmbeddings:
    return OpenAIEmbeddings(
        model = os.getenv("EMBEDDING_MODEL"),
        api_key = os.getenv("PERSONAL_OPENAI_KEY"),
        max_retries = 2
    )


def get_vector_store(collection_name: str) -> PGVector:
    embeddings = get_embeddings_client()
    return PGVector(
        embeddings = embeddings,
        collection_name = collection_name,
        connection = os.getenv("CONNECTION_STRING")
    )

def get_llm_client() -> ChatOpenAI:
    return ChatOpenAI(
        model = os.getenv("CHAT_MODEL"),
        temperature = 0,
        api_key = os.getenv("PERSONAL_OPENAI_KEY"),
        max_retries = 2
    )
