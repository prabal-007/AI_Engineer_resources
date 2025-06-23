from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate

load_dotenv()

llm = ChatOpenAI(model="gpt-4.1-nano")

template = "Write a {tone} email to {company} recruter, expressing intrest in the {position}," \
"mentioning {skill} as key strength. keep it to 4 lines max"

prompt_template = ChatPromptTemplate.from_template(template)

prompt = prompt_template.invoke({
    "tone": "formal",
    "company": "Google",
    "position": "AI Engineer",
    "skill": "AI"
})

# print(prompt)

result = llm.invoke(prompt)
print(result.content)