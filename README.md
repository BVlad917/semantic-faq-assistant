# Semantic FAQ Assistant
## Objective:
Develop a solution that can provide answers to users' questions by matching their queries with semantically similar questions in a predefined FAQ database. This system should further be enhanced to interact with the OpenAI API when it can't find a close enough match within the local FAQ database.

## Resources
- https://python.langchain.com/docs/get_started/introduction.html
- https://platform.openai.com/docs/guides/embeddings
- https://platform.openai.com/docs/guides/gpt/chat-completions-api
- https://fastapi.tiangolo.com/

## Requirements:
1. **Embedding Computing:** Use the provided Q&A examples (given below) to compute embeddings that will be used for similarity searches.
2. **Similarity Search:** When a user submits a query, search for the most similar question in your local FAQ database using the computed embeddings.
3. **Interacting with OpenAI API (via LangChain):** If the similarity score for the best local match is below a certain threshold, forward the question to the OpenAI API for an answer.
4. **FastAPI Endpoint:** Design an API endpoint using FastAPI where users can submit their questions and get answers.
5. **LangChain:** Use LangChain for any necessary natural language processing outside the scope of OpenAI.

## Challenge Steps:
1. Compute embeddings for each question in the FAQ database using the provided text and any necessary tools from LangChain.
2. Create a similarity search function that takes in a user's question, computes its embedding, and finds the most similar question from the FAQ database.
3. Design a decision function to decide whether the found match is close enough or if the question should be sent to OpenAI's API.
4. Implement the API endpoint using FastAPI where users can submit their question and retrieve an answer based on the logic from the previous steps.
5. Make sure that all interactions with the OpenAI API (via LangChain) are correctly handled and errors are gracefully managed.

## Example Input/Output:
#### Input
    POST /ask-question
    {
        "user_question": "How do I reset my account?"
    }

#### Output if a local match is found:
    {
        "source": "local",
        "matched_question": "How can I restore my account settings?",
        "answer": "Go to settings and click on 'restore default'."
    }

#### Output if forwarded to OpenAI API:
    {
        "source": "openai",
        "matched_question": "N/A",
        "answer": "To reset your account, typically, you'd navigate to account settings and look for the 'reset' option. However, specific instructions may vary based on the platform."
    }

#### Provided Text for Embedding Computing:
    faq_database = [
        {
            "question": "How do I change my profile information?",
            "answer": "Navigate to your profile page, click on 'Edit Profile', and make the desired changes."
        },
        {
            "question": "What steps do I take to reset my password?",
            "answer": "Go to account settings, select 'Change Password', enter your current password and then the new one. Confirm the new password and save the changes."
        },
        {
            "question": "How can I restore my account to its default settings?",
            "answer": "In the account settings, there should be an option labeled 'Restore Default'. Clicking this will revert all custom settings back to their original state."
        },
        {
            "question": "Is it possible to change my registered email address?",
            "answer": "Yes, navigate to account settings, find the 'Change Email' option, enter your new email, and follow the verification process."
        },
        {
            "question": "How can I retrieve lost data from my account?",
            "answer": "Contact our support team with details of the lost data. They'll guide you through the recovery process."
        },
        {
            "question": "Are there any guidelines on setting a strong password?",
            "answer": "Absolutely! Use a combination of uppercase and lowercase letters, numbers, and special characters. Avoid using easily guessable information like birthdays or names."
        },
        {
            "question": "Can I set up two-factor authentication for my account?",
            "answer": "Yes, in the security section of account settings, there's an option for two-factor authentication. Follow the setup instructions provided there."
        },
        {
            "question": "How do I deactivate my account?",
            "answer": "Under account settings, there's a 'Deactivate Account' option. Remember, this action is irreversible."
        },
        {
            "question": "What do I do if my account has been compromised?",
            "answer": "Immediately reset your password and contact our security team for further guidance."
        },
        {
            "question": "Can I customize the notifications I receive?",
            "answer": "Yes, head to the notifications settings in your account and choose which ones you'd like to receive."
        }
    ]
This FAQ database covers various common queries related to user accounts. The challenge would be to identify the most similar question to a user's query and return the corresponding answer. If no sufficiently similar question is found in the database, the query would be forwarded to the OpenAI API for a response.

## Evaluation Criteria
1. Quality of embeddings and similarity search results.
2. Correctness of the interaction with OpenAI API.
3. Robustness and error handling of the FastAPI application.
4. Env management.
5. Code structure, quality, and readability.

## Bonus Points
1. Authentication added to endpoints by using FastAPI’s dependency mechanism (`Depends(get_token)` - a hint)
2. PostgreSQL usage for info and embedding storage (`pgVector`)
3. Have the `postgres` instance run in `docker compose`
4. Scripts for managing embeddings DB: creating embeddings, updating embeddings (without deleting existing ones and being token-efficient), adding new collections
5. Create an AI Router using LangChain ([docs](https://python.langchain.com/v0.1/docs/use_cases/query_analysis/techniques/routing/))
    
    The router will allow us to handle 2 type of questions:
    a. is the question IT related and follows the topics that the system is designed to answer?
    b. if not, we have to route it to a Compliance Agent which will answer with a default output of 
    “This is not really what I was trained for, therefore I cannot answer. Try again.”
6. `Dockerfile` - run the system using a docker image, perhaps even `docker-compose.yaml`!
7. `Celery`-based async embeddings processing


### docker-compose commands
    # Start up our services
    docker-compose up --build -d
    
    # Add initial data
    docker-compose exec api python data_handling.py create --collection_name faq_collection --data_file faq_data.json
    
    # Update data
    docker-compose exec api python data_handling.py sync --collection_name faq_collection --data_file faq_data_reduced.json
    
    # Stop the services
    docker-compose down
