from langchain_core.prompts import ChatPromptTemplate

import langchain_utils
from models import RouteQuery


def get_router_chain():
    router_prompt = ChatPromptTemplate.from_template(
        """You are an expert at routing a user's question to the correct department.
    The IT department handles questions about account settings, password resets, profile information, and notifications.
    All other questions should be routed to Compliance.
    
    Based on the user's question, which route should it take?
    
    User Question:
    {question}
    """
    )
    llm = langchain_utils.get_llm_client()
    structured_llm = llm.with_structured_output(RouteQuery)
    router_chain = router_prompt | structured_llm
    return router_chain
