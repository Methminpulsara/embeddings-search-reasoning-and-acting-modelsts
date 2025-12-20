# pip install -qU langchain-pinecone
import os
from langchain.agents.middleware import HumanInTheLoopMiddleware
from langgraph.checkpoint.memory import InMemorySaver

from aiohttp.web_middlewares import middleware
from langchain.agents.middleware import HumanInTheLoopMiddleware
from langchain_pinecone import PineconeVectorStore
from pinecone import Pinecone

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
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from langchain_google_genai import ChatGoogleGenerativeAI
from pydantic import BaseModel
from typing import Optional
from fastapi import FastAPI

app = FastAPI(description="AI Agent for hr Rag System with Vector DATABASE")


class Request(BaseModel):

    question: str
    thread_id: str


class Responese(BaseModel):
    question: str
    anwser: str
    status: str
    thread_id:str
    message : Optional[str] = None


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

pc = Pinecone(api_key="VECTOR_DB_API_KEY")

vector_store = PineconeVectorStore(index_name="hr-pdf", embedding=embedding_model)

# -------------------------------------DN one meka embed karala save kara gnn
# vector_store = InMemoryVectorStore(embedding=embedding_model)
#
# # In memory storage ekk create kara gnnnw
vector_store.add_documents(documents=text_split)

OPENROUTER_KEY= os.getenv("OPENROUTER_KEY")
OPENROUTER_URL= os.getenv("OPENROUTER_URL")

# WADA KRNNE NTHTHM OPENROUTER API KEY EKA MAARU KARNN
model = ChatOpenAI(
    model="openai/gpt-oss-20b:free",
    api_key=OPENROUTER_KEY,
    base_url=OPENROUTER_URL
)


@tool
def enhance_query_user_query(user_query: str):
    """
    Enhances user query using an LLM for better retrieval

    Args:
        user_query: The user's raw query
    Returns:
        Enhanced user query optimized for semantic search
    """

    enhancement_prompt = ChatPromptTemplate.from_template(
        """
        You are a query optimization expert for an HR knowledge base.
        Your task is to expand users query to improve document retrieval

        Original Query: {original_query}

        Instruction: 
        - Expand abbreviations (PTO - Paid Time Off, HR - Human Resource)
        - Include relevant keywords for HR context
        - Maintain original intent

        Return only enhanced query nothing else
        """
    )

    enhance_chain = enhancement_prompt | model

    resp = enhance_chain.invoke({"original_query": user_query})

    return resp.content


@tool
def write_result_to_file(content: str, file_name: str = "query_result.txt"):
    """
    Write Query result to text file for persistent storage
    Args:
          content:The result content to write
          file_name:Target filename (default:query_result.txt)

    returns:
        Confirmation message of successful write

    """

    # try:
    with open(file_name, "a") as f:
        f.write(content)
    return f"successfuly wrote content to filename - {file_name} "
    # except Exception as e:
    #


@tool("content_and_artifact")
def retrieve_context(enhanced_user_query: str):
    """Retrive imfomations to help anwser user quaries"""
    retrived_doc = vector_store.similarity_search(enhanced_user_query, k=5)

    print(retrived_doc)

    contet = "\n\n".join(
        (f"Source : {doc.metadata} \n Content : {doc.page_content}")
        for doc in retrived_doc
    )
    return contet, retrived_doc


agent = create_agent(model=model,
                     tools=[retrieve_context, enhance_query_user_query, write_result_to_file],
                     system_prompt=""""
                            You have access to a tool that retrieves context from hr Use the tool to help to anwser the Questions.,
                            You have also access to a tool wich 
                            
                            """,
                     checkpointer=checkpointer,
                     middleware=[HumanInTheLoopMiddleware(
                         interrupt_on={
                             "retrieve_context":False,
                             "enhance_query_user_query": False,
                             "write_result_to_file":False
                         }
                     )]
                     )


@app.post("/quary", response_model=Responese)
def query(req: Request):

    config ={"configurable": {"theread_id ": req.thread_id}}

    try:
        resp = agent.invoke(
            {"messages": [{"role": "user", "content": req.question}]},
            config=config
        )

        final_answer = resp["messages"][-1].content

        return Responese(
            question=req.question,
            anwser=final_answer,
            status="success",
            thread_id=req.thread_id
        )
    except Exception as e:
        # Return the error message in the 'message' field
        return e


class ApproveRequest(BaseModel):
    thread_id : str
    decision : str

from langgraph.types import Command
@app.post('/approve')
def approve_and_continue(request:ApproveRequest):

    config ={"configurable": {"theread_id ": request.thread_id}}

    result = agent.invoke(
        Command(
            resume ={"decisions": [{"type" :request.decision}]}
        ),config
    )

    anwser = result["messages"][-1].content if result.get("messages") else "No Anwser"
    return anwser






# FILE ek hadenne nththte MODEL eke performence madhi nisa api key ekk gnn onh ekh nisa mekt