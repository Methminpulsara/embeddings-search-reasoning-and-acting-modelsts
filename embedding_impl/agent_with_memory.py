from dotenv import load_dotenv
from langgraph.checkpoint.memory import InMemorySaver
from langchain_ollama import ChatOllama
from langchain.agents import create_agent
from fastapi import  FastAPI
from pydantic import BaseModel


load_dotenv()

app = FastAPI(description="Agentic chat bot")

SYSTEM_PROMT = ("Your are a friendly AI Assistant"
                "Keep Conversations natural and helpful")

chchkpointer = InMemorySaver()

model = ChatOllama(model="llama3.2:3b")

agent = create_agent(model=model , system_prompt=SYSTEM_PROMT , tools=[], checkpointer=chchkpointer)


class ChatRequest(BaseModel):

    user_id: str
    message: str


class ChatResponse(BaseModel):
    reply: str



@app.post("/", response_model=ChatResponse)
def chat(req: ChatRequest):
    config = {"configurable": {"thread_id": req.user_id}}  # use user_id for thread_id

    result = agent.invoke(
        {
            "messages": [{"role": "user", "content": req.message}]
        },
        config=config   # pass the config here
    )
    return ChatResponse(reply=result["messages"][-1].content)
