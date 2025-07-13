import os
import logging
from typing import Annotated
from dotenv import load_dotenv
from fastapi.params import Depends
from fastapi import FastAPI, Header, HTTPException

from celery_worker import process_new_faq
from rag_chains.system_chain import get_system_chain
from langchain_community.utilities import SQLDatabase

from models import AnswerResponse, QuestionRequest, NewFAQResponse, NewFAQRequest


load_dotenv()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI()
system_chain = get_system_chain()


keys_string = os.getenv("VALID_API_KEYS")
VALID_API_KEYS = keys_string.split(',')


def get_token(x_api_key: Annotated[str, Header()]):
    """
    Dependency that checks for a valid X-API-Key header.
    If the key is missing or invalid, it raises an HTTPException,
    stopping the request from proceeding.
    """
    if x_api_key not in VALID_API_KEYS:
        raise HTTPException(status_code=403, detail="Invalid API Key")


@app.post("/ask", response_model=AnswerResponse)
def ask_question(request: QuestionRequest, token: str = Depends(get_token)):
    try:
        logger.info(f"Invoking RAG chain for question: {request.question[:50]}...")
        result = system_chain.invoke({"question": request.question})
        return result
    except Exception as e:
        logger.error(f"Error during RAG chain invocation: {e}", exc_info=True)
        raise HTTPException(status_code=503, detail="Service unavailable: Could not process the question.")


@app.post("/add_faq", response_model=NewFAQResponse)
def add_faq(request: NewFAQRequest, token: str = Depends(get_token)):
    """ Accepts a new FAQ and queues it for background processing. """
    try:
        logger.info(f"Queuing new FAQ: {request.question[:50]}...")
        task = process_new_faq.delay(collection_name=os.getenv("COLLECTION_NAME"), question=request.question, answer=request.answer)
        return {
            "message": "FAQ received and is being processed in the background.",
            "task_id": task.id
        }
    except Exception as e:
        logger.error(f"Error queueing FAQ task: {e}", exc_info=True)
        raise HTTPException(status_code=503, detail="Service unavailable: Could not queue the FAQ for processing.")


@app.get("/")
def get_all_qas():
    """ For testing purposes. """
    db = SQLDatabase.from_uri(os.getenv("CONNECTION_STRING"))
    results = db.run("SELECT langchain_pg_embedding.cmetadata FROM langchain_pg_embedding;")
    if len(results.strip()) == 0:
        return []
    results = eval(results)
    results = [result[0] for result in results]
    return results
