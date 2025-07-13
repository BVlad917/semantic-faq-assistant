import os
import logging
from celery import Celery
from dotenv import load_dotenv

from langchain_postgres import PGVector
from langchain_core.documents import Document
from langchain_openai import OpenAIEmbeddings

import langchain_utils

load_dotenv()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

celery_app = Celery("tasks", broker=os.getenv("CELERY_BROKER_URL"), backend=os.getenv("CELERY_RESULT_BACKEND"))


@celery_app.task(
    name="process_new_faq",
    bind=True,
    autoretry_for=(Exception,),
    retry_kwargs={'max_retries': 3},
    retry_backoff=True,
    retry_backoff_max=60
)
def process_new_faq(self, collection_name: str, question: str, answer: str):
    """ Celery task to compute embedding for a new FAQ and add it to the vector store. """
    try:
        logger.info(f"[Task ID: {self.request.id}] Processing FAQ: {question}")
        vector_store = langchain_utils.get_vector_store(collection_name)
        new_document = Document(
            page_content=f"Question: {question}\nAnswer: {answer}",
            metadata={
                'original_question': question,
                'original_answer': answer
            }
        )
        vector_store.add_documents([new_document])
        logger.info(f"[Task ID: {self.request.id}] Successfully added FAQ to vector store.")

    except Exception as e:
        logger.error(f"[Task ID: {self.request.id}] Failed to process FAQ after multiple retries: {e}", exc_info=True)
        raise
