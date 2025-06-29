from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain.schema.output_parser import StrOutputParser

load_dotenv()
model = ChatOpenAI(model="gpt-4.1-nano")

prompt_template = ChatPromptTemplate.from_messages(
    [
        ("system", "You are a facts expert who knows facts about {sport}"),
        ("human", "Tell me {count} facts {subject}."),
    ]
)

chain = prompt_template | model | StrOutputParser()

result = chain.invoke(
    {
        "sport": "Football",
        "count": 2,
        "subject": "Messi Vs Ronaldo"
    }
)

print(result)