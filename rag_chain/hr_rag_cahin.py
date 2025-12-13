# pip install -U langchain-text-splitters
# pip install pypdf

# wada kare nthi unoth rag llm eka OPENAPI AI KEY ekk setup krnn


from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from dotenv import load_dotenv
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_ollama import OllamaEmbeddings
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate

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

rag_llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash")


def retrieve_context(user_query: str):
    retrived_doc = vector_store.similarity_search(user_query, k=5)

    print(retrived_doc)

    contet = "\n\n".join(
        (f"Source : {doc.metadata} \n Content : {doc.page_content}")
        for doc in retrived_doc
    )
    return contet

prompt_template = ChatPromptTemplate.from_messages([
        ("system", """You are a Help full Assistant who provides news using the provided context.
                    Use only the information from the context to answer.
                    If context have the answer sya so"""),
        ("human", "Context : \n {context} \n\n Question :{question}")
    ])



    # CHAIN EKK Hadana widiyat


rag_chain = prompt_template | rag_llm
while True:
    q = input("Enter Your Question : ")
    resp = rag_chain.invoke({"context":  retrieve_context(q) , "question":q})
    print(resp)


retrieve_context("What is the promotion policy")
