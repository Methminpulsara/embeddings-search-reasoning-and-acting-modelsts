import os
import requests
from typing import List
from dotenv import load_dotenv
from fastapi import FastAPI
from langchain_classic.chains.llm import LLMChain
from langchain_core.prompts import PromptTemplate
from pydantic import BaseModel
from langchain_core.messages import HumanMessage
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from dotenv import load_dotenv
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_ollama import OllamaEmbeddings
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate



from langgraph.checkpoint.memory import InMemorySaver
from langchain_ollama import ChatOllama
from langchain.agents import create_agent
from langchain_core.tools import  tool

from langchain_google_genai import ChatGoogleGenerativeAI

checkpointer = InMemorySaver()

load_dotenv()

# embedding_model = GoogleGenerativeAIEmbeddings(model="models/gemini-embedding-001")
embedding_model = OllamaEmbeddings(model="nomic-embed-text")

text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)

# ------------------- doc ek load krla read krnw
filepath = "./hr_manual.pdf"
# loader = PyPDFLoader(file_path=filepath) Thani loku page ek ookma ekt
loader = PyPDFLoader(file_path=filepath, mode="page")

# print(loader.load())


text_split = text_splitter.split_documents(loader.load())

# -------------------------------------DN one meka embed karala save kara gnn

vector_store = InMemoryVectorStore(embedding=embedding_model)

# In memory storage ekk create kara gnnnw
vector_store.add_documents(documents=text_split)

model = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash"
)


@tool("content_and_artifact")
def retrieve_context(user_query: str):
    """Retrive imfomations to help anwser user quaries"""
    retrived_doc = vector_store.similarity_search(user_query, k=5)

    print(retrived_doc)

    contet = "\n\n".join(
        (f"Source : {doc.metadata} \n Content : {doc.page_content}")
        for doc in retrived_doc
    )
    return contet , retrived_doc

agent = create_agent(model=model ,
                     tools=[retrieve_context] ,
                     system_prompt=""""
                            Tou have access to a tool that retrieves context from hr Use the tool to help to anwser the Question.
                            """
                     )
query = input("Enter your Question :")

resp = agent.invoke({
    "messages" : [{"role":"user", "content": query}]
})

print(resp["messages"][-1].content[0]["text"])