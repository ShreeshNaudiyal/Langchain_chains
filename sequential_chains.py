from langchain_groq import ChatGroq
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser

load_dotenv()

# Prompt 1
prompt1 = PromptTemplate(
    template="Generate a detailed report on {topic}",
    input_variables=["topic"]
)

# Prompt 2
prompt2 = PromptTemplate(
    template="Generate a 5-point summary from the following text:\n\n{text}",
    input_variables=["text"]
)

# Groq LLM
model = ChatGroq(
    model="llama-3.3-70b-versatile",
    temperature=0.5
)

# Output parser
parser = StrOutputParser()

# Chain
chain = (
    prompt1
    | model
    | parser
    | prompt2
    | model
    | parser
)

# Invoke
result = chain.invoke({"topic": "Unemployment in India"})

print(result)

# Print chain graph
chain.get_graph().print_ascii()