from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from dotenv import load_dotenv

load_dotenv()

llm = ChatOpenAI(model="gpt-4.1-nano")

messages = [
    SystemMessage("You are a Gen AI expert"),
    HumanMessage("What is Agentic AI")
]

result = llm.invoke(messages)

print(result.content)