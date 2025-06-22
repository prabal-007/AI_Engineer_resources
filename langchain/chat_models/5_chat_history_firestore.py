from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from google.cloud import firestore
from langchain_google_firestore import FirestoreChatMessageHistory

load_dotenv()

PROJECT_ID="learning-projects-7728f"
COLLECTION_ID="chat_history"
SESSION_ID="test_session_id"

print("Initilize firestore client")
client = firestore.Client(project=PROJECT_ID)

print("Initilizze firestore chat message history")
chat_history = FirestoreChatMessageHistory(
    session_id=SESSION_ID,
    collection=COLLECTION_ID,
    client=client
)

print("Chat history initilized")
print("Curremt chat history: ", chat_history)

model = ChatOpenAI(model="gpt-4.1-nano")

print("Start chatting with AI, type 'exit' to quit.")

while True:
    human_input = input("You: ")
    if human_input.lower() == "exit":
        break
    chat_history.add_user_message(human_input)

    ai_response = model.invoke(chat_history.messages)
    chat_history.add_ai_message(ai_response.content)

    print("AI: ", ai_response.content)