from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate

load_dotenv

llm = ChatOpenAI(model="gpt-4.1-nano")

# template = "Write a mail to {company} hiring manager to show intrest in {position} position, with {skill} skills as strength. Keep it 4 lines maximum"

messages = [
    ("system", "You are a professional email writer, who helps in writing cold emails for {role}."),
    ("human", "Write a mail to {company} hiring manager to show intrest in {position} position, with {skill} skills as strength. Keep it 4 lines maximum")
]

# prompt_template = ChatPromptTemplate.from_template(template)
message_template = ChatPromptTemplate.from_messages(messages)
# what is it about 
prompt = message_template.invoke({
    "role": "Job search",
    "company": "Tesla",
    "position": "AI Engineer",
    "skill": "AI"
})

# print(prompt)

result = llm.invoke(prompt)
print(result.content)