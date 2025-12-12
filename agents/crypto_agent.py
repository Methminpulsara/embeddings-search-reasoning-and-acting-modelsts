# agent_crypto_app.py
import os
import requests
from typing import List
from dotenv import load_dotenv
from fastapi import FastAPI
from langchain_classic.chains.llm import LLMChain
from langchain_core.prompts import PromptTemplate
from pydantic import BaseModel
from fastapi.responses import JSONResponse
from langchain_core.messages import HumanMessage


from langgraph.checkpoint.memory import InMemorySaver
from langchain_ollama import ChatOllama
from langchain.agents import create_agent
from langchain_core.tools import  tool

load_dotenv()

# -------------------------------
# Environment variables
# -------------------------------
COINGECKO_URL = os.getenv("COINGECKO_URL", "https://api.coingecko.com/api/v3/coins/markets")
MODEL_NAME = "llama3.2:3b"

# -------------------------------
# FastAPI app
# -------------------------------
app = FastAPI(title="Crypto Analysis AI Tool Agent")

# -------------------------------
# Memory / Checkpointer
# -------------------------------
checkpointer = InMemorySaver()

# -------------------------------
# Model
# -------------------------------
model = ChatOllama(model="llama3.2:3b")



# -------------------------------
# Tools
# -------------------------------
@tool
def get_crypto_insights(coin_list: List[str]):
    """Fetch crypto market data from Coingecko API"""
    params = {"vs_currency": "usd", "ids": ",".join(coin_list)}
    resp = requests.get(COINGECKO_URL, params=params)
    resp.raise_for_status()
    return resp.json()


@tool
def analyze_crypto_data(raw_market_data: str):
    """Analyze crypto market data using LLM and return JSON response"""
    SYSTEM_PROMPT = """
    You are a Crypto Analyst AI. Analyze the given crypto market data and 
    determine which coin has the strongest potential. 
    Output ONLY valid JSON in the following format:

    {{
        "comparison": {{
            "winner": "<coin name with strongest outlook>",
            "summary": "<1-2 sentence human summary>",
            "reasons": ["<reason1>", "<reason2>", "<reason3>"]
        }}
    }}

    Market data:
    {market_data}
    """

    prompt = PromptTemplate(
        input_variables=["market_data"],
        template=SYSTEM_PROMPT
    )

    chain = LLMChain(llm=model, prompt=prompt)
    return chain.run({"market_data": raw_market_data})


# -------------------------------
# Create agent
# -------------------------------
agent = create_agent(
    model=model,
    tools=[get_crypto_insights, analyze_crypto_data],
    system_prompt="You are a friendly AI assistant. Keep conversations natural and helpful.",
    checkpointer=checkpointer
)


# -------------------------------
# Request / Response Models
# -------------------------------
class CryptoRequest(BaseModel):
    user_id: str
    coin_list: List[str]


class CryptoResponse(BaseModel):
    reply: str


# -------------------------------
# API Route
# -------------------------------
@app.post("/crypto/analysis", response_model=CryptoResponse)
def analysis(req: CryptoRequest):
    # Create thread_id from user_id for memory
    config = {"configurable": {"thread_id": req.user_id}}

    # Ask agent to analyze coins
    result = agent.invoke(
        {
            "messages": [HumanMessage(content=f"Analyze crypto currencies: {','.join(req.coin_list)}")]
        },
        config=config
    )

    # Return last message content
    return CryptoResponse(reply=result["messages"][-1].content)
