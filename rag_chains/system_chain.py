from langchain_core.runnables import RunnableLambda, RunnablePassthrough, RunnableBranch

from models import Route
from rag_chains.it_chain import get_it_chain
from rag_chains.router_chain import get_router_chain
from rag_chains.compliance_chain import get_compliance_chain


def get_system_chain():
    it_chain = get_it_chain()
    compliance_chain = get_compliance_chain()
    router_chain = get_router_chain()

    it_path = RunnableLambda(lambda x: x["input"]) | it_chain
    branch = RunnableBranch(
        (lambda x: x["decision"].route == Route.IT, it_path),
        compliance_chain,
    )

    full_chain = {
        "decision": router_chain,
        "input": RunnablePassthrough()
    } | branch
    return full_chain
