from langchain.agents import create_agent
from langchain_ollama import ChatOllama
from langchain_core.tools import tool
from langchain_core.messages import HumanMessage

model = ChatOllama(model="llama3.2:3b")


@tool
def add(x: int, y: int):
    """Adds two numbers"""
    return x + y


@tool
def subtract(x: int, y: int):
    """Subtract two numbers"""
    return x - y


tools = [add, subtract]

agent = create_agent(
    model=model,
    tools=tools
)

resp = agent.invoke({
    "messages": [HumanMessage(content="whta is 10  + 20")]
})

print(resp)

for msg in resp["messages"]:
    print(f"{msg}")

