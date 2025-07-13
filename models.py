from enum import Enum
from pydantic import BaseModel, Field


### FastAPI models ###
class QuestionRequest(BaseModel):
    """The request model for a user's question."""
    question: str

class AnswerResponse(BaseModel):
    """The response model for the agent's answer."""
    source: str
    matched_question: str
    answer: str

class NewFAQRequest(BaseModel):
    """Request model for adding a new FAQ document."""
    question: str
    answer: str

class NewFAQResponse(BaseModel):
    """Response model after submitting a new FAQ for processing."""
    message: str
    task_id: str


### Chain models ###
class Route(str, Enum):
    """ The available routes for the user's query. """
    IT = "IT"
    COMPLIANCE = "COMPLIANCE"


class RouteQuery(BaseModel):
    """A model to structure the output of our routing LLM."""
    route: Route = Field(
        ...,
        description="The route to take for the user's query, either 'it' or 'compliance'."
    )
