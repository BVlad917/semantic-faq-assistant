import hashlib
import os
import json
import argparse
from sqlalchemy import create_engine, text

from dotenv import load_dotenv
from langchain.docstore.document import Document

import langchain_utils

load_dotenv()


def create_collection(collection_name: str, faq_data_filepath: str = "./faq_data.json"):
    """ Read FAQs from file -> Embed them -> Connect to the vector DB -> Store the documents. """
    with open(faq_data_filepath) as fp:
        faq_dicts = json.load(fp)

    documents, ids = [], []
    for faq in faq_dicts:
        question, answer = faq["question"], faq["answer"]
        doc_id = hashlib.sha256(question.encode()).hexdigest()  # the question determines the unique ID

        documents.append(Document(
            page_content = f"Question: {faq['question']}\nAnswer: {faq['answer']}",
            metadata = {'original_question': faq['question'], 'original_answer': faq['answer']}
        ))
        ids.append(doc_id)

    vector_store = langchain_utils.get_vector_store(collection_name=collection_name)
    vector_store.add_documents(documents, ids=ids, pre_delete_collection=True)


def sync_collection(collection_name: str, faq_data_filepath: str):
    """
    Synchronizes the vector DB with a local JSON file.
    - Add new FAQs.
    - Delete outdated FAQs.
    - Update changed FAQs.
    """
    with open(faq_data_filepath) as fp:
        faq_dicts = json.load(fp)

    # What are the docs we want in our collection in the end?
    desired_docs = {}
    for qa in faq_dicts:
        question = qa['question']
        answer = qa['answer']
        doc_id = hashlib.sha256(question.encode()).hexdigest()  # the question determines the unique ID

        desired_docs[doc_id] = Document(
            page_content=f"Question: {question}\nAnswer: {answer}",
            metadata={'original_question': question, 'original_answer': answer}
        )

    # What are the current docs in our collection?
    engine = create_engine(os.getenv("CONNECTION_STRING"))
    with engine.connect() as conn:
        # The SQL query remains the same
        results = conn.execute(text(
            f'SELECT c.id, c.document, c.cmetadata FROM langchain_pg_embedding c '
            f'JOIN langchain_pg_collection coll ON c.collection_id = coll.uuid '
            f'WHERE coll.name = :collection_name'
        ), {'collection_name': collection_name}).fetchall()
    current_docs = {str(row[0]): {'page_content': row[1], 'metadata': row[2]} for row in results}

    # Find the differences and resolve them
    desired_ids = set(desired_docs.keys())
    current_ids = set(current_docs.keys())

    ids_to_add = list(desired_ids - current_ids)
    ids_to_delete = list(current_ids - desired_ids)
    ids_to_check = list(desired_ids.intersection(current_ids))

    docs_to_add = [desired_docs[id_] for id_ in ids_to_add]

    # For the documents which existed before and exist now (identified by the ID generated from the question)
    # check if their content (the answer) has changed
    for id_ in ids_to_check:
        if desired_docs[id_].page_content != current_docs[id_]['page_content']:
            ids_to_delete.append(id_)
            docs_to_add.append(desired_docs[id_])
            ids_to_add.append(id_)

    vector_store = langchain_utils.get_vector_store(collection_name)

    if ids_to_delete:
        vector_store.delete(ids=ids_to_delete)

    if docs_to_add:
        vector_store.add_documents(docs_to_add, ids=ids_to_add)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Manage PGVector embeddings database.")
    subparsers = parser.add_subparsers(dest='command', required=True)

    # Sub-parser for the 'create' command
    create_parser = subparsers.add_parser('create', help='DESTRUCTIVE. Create a new collection and embed documents.')
    create_parser.add_argument('--collection_name', type=str, required=True, help='Name of the collection.')
    create_parser.add_argument('--data_file', type=str, required=True, help='Path to the JSON file with Q&A data.')

    # Sub-parser for the 'add' command
    add_parser = subparsers.add_parser('sync', help='Synchronize a collection with a source file (add, update, delete).')
    add_parser.add_argument('--collection_name', type=str, required=True, help='Name of the collection to sync.')
    add_parser.add_argument('--data_file', type=str, required=True, help='Path to the master JSON file (source of truth).')

    args = parser.parse_args()
    if args.command == 'create':
        create_collection(args.collection_name, args.data_file)
    elif args.command == 'sync':
        sync_collection(args.collection_name, args.data_file)
    else:
        raise ValueError(f'Unknown command: {args.command}')
