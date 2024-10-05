# Script to query to Elasticsearch from the text in natural language using LLM

import json
import requests
import csv
from datetime import datetime
import dotenv
import os

from typing import List, Tuple
import argparse

from langchain.output_parsers import PydanticOutputParser
from langchain_core.prompts import PromptTemplate
# from langchain_core.pydantic_v1 import BaseModel, Field
from pydantic import BaseModel, Field

from langchain_openai import ChatOpenAI

dotenv.load_dotenv()


def create_query(mapping: str, text: str):
    chat_model = ChatOpenAI(
        model="gpt-4o",
        temperature=0.7,
        max_retries=3
    )

    class EvaluationResult(BaseModel):
        query: str = Field(description="The query in Elasticsearch DSL format.")

    query = f"""Given the mapping delimited by triple backticks 
    ```{mapping}```
    translate the text delimited by triple quotes in a valid Elasticsearch DSL query `{text}`.
    Give me only the json code part of the answer. Compress the json output removing spaces.
    """

    parser = PydanticOutputParser(pydantic_object=EvaluationResult)

    prompt = PromptTemplate(
        template="Answer the user query.\n{format_instructions}\n{query}\n",
        input_variables=["query"],
        partial_variables={"format_instructions": parser.get_format_instructions()},
    )

    chain = prompt | chat_model | parser

    evaluation_result = chain.invoke({"query": query})
    return evaluation_result.query 


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Query to Elasticsearch from the text in natural language using LLM")
    parser.add_argument("--mapping-file", required=True, type=str, help="The mapping file in Elasticsearch DSL format")
    parser.add_argument("--text", required=True, type=str, help="The text in natural language")
    args = parser.parse_args()
    print("args: ", args)

    mapping_definition = open(args.mapping_file, "r").read()

    query = create_query(mapping_definition, args.text)
    print("---\nQuery:\n")
    print(query)

    # Query to Elasticsearch
    url = "http://localhost:9200/bioproject/_search/"
    headers = {
        "Content-Type": "application/json"
    }
    response = requests.get(url, headers=headers, data=query)
    print("---\nResults:\n")
    print(response.text)