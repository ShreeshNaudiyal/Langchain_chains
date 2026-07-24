from langchain_groq import ChatGroq
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser, PydanticOutputParser
from langchain.schema.runnable import RunnableBranch, RunnableLambda
from pydantic import BaseModel, Field
from typing import Literal

load_dotenv()

# Initialize Groq model
model = ChatGroq(
    model="llama-3.3-70b-versatile",
    temperature=0.5
)

parser = StrOutputParser()

class Feedback(BaseModel):
    sentiment: Literal["positive", "negative"] = Field(
        description="Give the sentiment of the feedback"
    )

parser2 = PydanticOutputParser(pydantic_object=Feedback)

prompt1 = PromptTemplate(
    template="""
Classify the sentiment of the following feedback text into positive or negative.

Feedback:
{feedback}

{format_instruction}
""",
    input_variables=["feedback"],
    partial_variables={
        "format_instruction": parser2.get_format_instructions()
    },
)

classifier_chain = prompt1 | model | parser2

prompt2 = PromptTemplate(
    template="Write an appropriate response to this positive feedback.\n\n{feedback}",
    input_variables=["feedback"],
)

prompt3 = PromptTemplate(
    template="Write an appropriate response to this negative feedback.\n\n{feedback}",
    input_variables=["feedback"],
)

branch_chain = RunnableBranch(
    (
        lambda x: x.sentiment == "positive",
        prompt2 | model | parser,
    ),
    (
        lambda x: x.sentiment == "negative",
        prompt3 | model | parser,
    ),
    RunnableLambda(lambda _: "Could not determine sentiment"),
)

chain = classifier_chain | branch_chain

response = chain.invoke(
    {"feedback": "This is a beautiful phone"}
)

print(response)
