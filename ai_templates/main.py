import argparse
import nbformat as nbf
from typing import Optional


def create_notebook(
    filename: str, db: Optional[bool] = False, asnc: Optional[bool] = False
):
    """Creates a notebook using nbformat and populate the first 3 cells

    Args:
        filename (str): The name of the notebook
        db (bool, optional): Whether to use a vector database. Defaults to False.
        asnc (bool, optional): Whether to use async chat completion. Defaults to False.

    Returns:
        None
    """

    nb = nbf.v4.new_notebook()

    db1_content = """
from dotenv import load_dotenv
import os
import chromadb
from chromadb.utils import embedding_functions
chroma_client = chromadb.Client()

load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

if OPENAI_API_KEY is None:
    raise Exception("API key not found in environment variables")

embedder = embedding_functions.OpenAIEmbeddingFunction(
    api_key=OPENAI_API_KEY,
    model_name="text-embedding-ada-002",
)

collection = chroma_client.create_collection(name="", embedding_function=embedder)"""

    # Create first cell
    cell1_content = f"""\
from ai_templates.oai.responses. {"async_chat_response import async_create_chat_completion" if asnc else "chat_response import create_chat_completion"}
from ai_templates.oai.types.base import ChatSequence, Message{", ChatMessages" if asnc else ""}
{db1_content if db else ""}
    """
    cell1 = nbf.v4.new_code_cell(cell1_content)

    # Create second cell
    cell2_content = f"""system_prompt=""
system_prompt_msg = Message("system", system_prompt)

description = ""
description_msg = Message("system", description)

user_query=""
user_query_msg = Message("user", user_query)


chat_sequence = ChatSequence([system_prompt_msg, description_msg, user_query_msg])
{"chat_sequence2 = ChatSequence([description_msg, user_query_msg])" if asnc else ""}"""

    cell2 = nbf.v4.new_code_cell(cell2_content)

    db3_content = f"""
collection.add(
    documents={"async_response1" if asnc else "[response1]"},
    metadatas=[{{"": ""}}],
    ids=[""]
)"""

    cell3_content = f"""\
{"async_response1 = await async_create_chat_completion(ChatMessages([chat_sequence, chat_sequence2]), return_every=True, response_list=response_list)" if asnc else "response1 = create_chat_completion(chat_sequence)"}
{db3_content if db else ""}
{"async_response1" if asnc else "response1"}"""

    cell3 = nbf.v4.new_code_cell(cell3_content)

    # Add cells to notebook
    nb.cells = [cell1, cell2, cell3]

    with open(filename, "w") as f:
        nbf.write(nb, f)


def main():
    parser = argparse.ArgumentParser(description="Generate a new Jupyter notebook")
    parser.add_argument("filename", help="Name of the notebook to create")
    parser.add_argument(
        "-db", action="store_true", help="With or without a vector database"
    )
    parser.add_argument(
        "-asnc",
        action="store_true",
        help="Async chat completion instead of a regular one",
    )
    args = parser.parse_args()

    create_notebook(f"{args.filename}.ipynb", args.db, args.asnc)


if __name__ == "__main__":
    main()
