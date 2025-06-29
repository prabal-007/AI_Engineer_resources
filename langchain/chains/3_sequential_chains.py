from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain.schema.runnable import RunnableLambda, RunnablePassthrough
from langchain.schema.output_parser import StrOutputParser

load_dotenv()

model = ChatOpenAI(model="gpt-4.1-nano")

fact_template = ChatPromptTemplate.from_messages([
    ("system", "You are an expert in {topic}."),
    ("human", "Tell me {count} facts about {subject}.")
])

translator_prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a language translator."),
    ("human", "Convert the following text to {language}: {text}")
])

to_translator_input = RunnableLambda(lambda inputs: {
    "text": inputs["facts"],
    "language": inputs["language"]
})

chain = (
    {
        "facts": fact_template | model | StrOutputParser(),
        "language": RunnablePassthrough()
    }
    | to_translator_input
    | translator_prompt
    | model
    | StrOutputParser()
)

response = chain.invoke({
    "topic": "India",
    "count": 2,
    "subject": "cricket",
    "language": "French"
})

print(response)