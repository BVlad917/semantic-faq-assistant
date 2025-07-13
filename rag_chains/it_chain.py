import os
import logging
from dotenv import load_dotenv

from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableLambda
from langchain.prompts import ChatPromptTemplate

import langchain_utils

load_dotenv()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# 1. FAQ Sub-Chain inside IT Chain
vector_store = langchain_utils.get_vector_store(os.getenv("COLLECTION_NAME"))
faq_chain = RunnableLambda(lambda info: {
    "source": "local",
    "matched_question": info["retrieved_doc"][0][0].metadata["original_question"],
    "answer": info["retrieved_doc"][0][0].metadata["original_answer"]
})

# 2. LLM Sub-Chain inside IT Chain
llm = langchain_utils.get_llm_client()
llm_prompt = ChatPromptTemplate.from_template("Answer the following question based on general knowledge: {question}")
llm_chain = (
    llm_prompt
    | llm
    | StrOutputParser()
    | RunnableLambda(lambda answer_str: {
        "source": "openai",
        "matched_question": "N/A",
        "answer": answer_str
    })
)


# Combine FAQ and LLM sub-rag_chains to create the IT chain
def it_route_rag(info):
    if info["retrieved_doc"] is None:
        logging.info("ROUTE: No retrieved document found. Falling back to LLM.")
        return llm_chain

    doc, cos_dist = info["retrieved_doc"][0]
    if cos_dist < float(os.getenv("MAX_COSINE_DIST")):
        logging.info(f"ROUTE: Found a good match with cosine distance {cos_dist:.4f}. Using FAQ.")
        return faq_chain
    else:
        logging.info(f"ROUTE: Cosine distance {cos_dist:.4f} is too high. Falling back to LLM.")
        return llm_chain


def get_it_chain():
    it_chain_input = {
        "question": lambda x: x["question"],
        "retrieved_doc": lambda x: vector_store.similarity_search_with_score(x["question"], k=1)
    }
    it_chain = it_chain_input | RunnableLambda(it_route_rag)
    return it_chain
