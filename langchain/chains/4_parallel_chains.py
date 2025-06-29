from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain.schema.output_parser import StrOutputParser
from langchain.schema.runnable import RunnableLambda, RunnableParallel

load_dotenv()

model = ChatOpenAI(model="gpt-4.1-nano")

summary_template = ChatPromptTemplate.from_messages([
    ("system", "Your are a expert social media creative write at a marketing company."),
    ("human", "Crete social media post for topic {topic}.")
])

def for_linkedin(post):
    linkedin_template = ChatPromptTemplate.from_messages([
        ("system", "You are an expert social media content write."),
        ("human", "Crete a linkedin post for topic {post}")
    ])
    return linkedin_template.format_prompt(post=post)

def for_x(post):
    x_template = ChatPromptTemplate.from_messages([
        ("system", "You are an expert social media content writer"),
        ("human", "Crete a X/twitter post for topic {post}")
    ])
    return x_template.format_prompt(post=post)

def combine_results(linkedin_post, x_post):
    return f"Linkedin Post:\n {linkedin_post}\n\nX Post:\n {x_post}"

linkedin_post_chain = RunnableLambda(lambda x: for_linkedin(x)) | model | StrOutputParser()

x_post_chain = RunnableLambda(lambda x: for_x(x)) | model | StrOutputParser()

chain = (
    summary_template
    | model
    | StrOutputParser()
    | RunnableParallel(branches={"linkedin": linkedin_post_chain, "x": x_post_chain})
    | RunnableLambda(lambda x: combine_results(x["branches"]["linkedin"], x["branches"]["x"]))
)

response = chain.invoke({
    "topic": "Virat Kolhi retired from test cricket"
    }
)

print(response)