from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain.schema.output_parser import StrOutputParser
from langchain.schema.runnable import RunnableLambda, RunnableBranch

load_dotenv()

model = ChatOpenAI(model="gpt-4.1-nano")

positive_feedback_template = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful assistent."),
    ("human", "Generate a thank you note for the positive feedback: {feedback}.")
])

negative_feedback_template = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful assistent"),
    ("human", "Generate a response addrerssing this negative feedback: {feedback}.")
])

neutral_feedback_template = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful assistent."),
    ("human", "Generate a response to request more detail for this neutral feedback: {feedback}")
])

escalate_feedback_template = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful assistent"),
    ("human", "Generate a message to esclate this feedback to human agent: {feedback}")
])

# sentimant classification prompt
classification_template = ChatPromptTemplate.from_messages([
    ("system", "You are a expert sentiment classifier agent."),
    ("human", "Classify the sentiment og this feedback as positive, negative, neutral or esclate: {feedback}.")
])

branches = RunnableBranch(
    (
        lambda x: "positive" in x,
        positive_feedback_template | model | StrOutputParser()
    ),
    (
        lambda x: "negative" in x,
        negative_feedback_template | model | StrOutputParser()
    ),
    (
        lambda x: "neutral" in x,
        neutral_feedback_template | model | StrOutputParser()
    ),
    escalate_feedback_template | model | StrOutputParser()
)

classification_chain = classification_template | model | StrOutputParser()

chain = classification_chain | branches

# example review
# Good review - "The product is excellent. I really enjoyed using it and found it very helpful."
# Bad review - "The product is terrible. It broke after just one use and the quality is very poor."
# Neutral review - "The product is okay. It works as expected but nothing exceptional."
# Default - "I'm not sure about the product yet. Can you tell me more about its features and benefits?"

# review = "The product is excellent. I really enjoyed using it and found it very helpful."
# review = "The product is terrible. It broke after just one use and the quality is very poor."
# review = "The product is okay. It works as expected but nothing exceptional."
review = "this is a trrible product, i want refund."
# review = "I'm not sure about the product yet. Can you tell me more about its features and benefits?"

response = chain.invoke(
    {
        "feedback": review
    }
)

print(response)