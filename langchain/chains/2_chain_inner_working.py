from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain.schema.output_parser import StrOutputParser
from langchain.schema.runnable import RunnableLambda, RunnableSequence

load_dotenv()

model = ChatOpenAI(model="gpt-4.1-nano")

prompt_template = ChatPromptTemplate.from_messages(
    [
        ("system", "You are fects expert in {topic}"),
        ("human", "Tell me {count} facts.")
    ]
)


# chain = prompt_template | model | StrOutputParser()

# this is waht happens under the hood when above statement is used

format_prompt = RunnableLambda(lambda x: prompt_template.format_prompt(**x))
invoke_model = RunnableLambda(lambda x: model.invoke(x.to_messages()))
parse_output = RunnableLambda(lambda x: x.content)

chain = RunnableSequence(first=format_prompt, middle=[invoke_model], last=parse_output)


response = chain.invoke({
    "topic": "India",
    "count": 2
})

print(response)