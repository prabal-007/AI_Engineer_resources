from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage

load_dotenv()
llm = ChatOpenAI(model="gpt-4.1-nano")
chat_history = []

system_message = SystemMessage(content="You are a AI tuitor, with experties in GenAI")
chat_history.append(system_message)

while True:
    query = input("You: ")
    if query.lower() == "exit":
        break
    
    chat_history.append(HumanMessage(content=query))
    result = llm.invoke(chat_history)
    response = result.content
    chat_history.append(AIMessage(content=response))

    print("AI: ", response[:300])
