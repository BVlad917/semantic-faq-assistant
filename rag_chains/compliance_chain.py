from langchain_core.runnables import RunnableLambda


def get_compliance_chain():
    compliance_chain = RunnableLambda(lambda x: {
        "source": "compliance_policy",
        "matched_question": "N/A",
        "answer": "This is not really what I was trained for, therefore I cannot answer. Try again."
    })
    return compliance_chain
