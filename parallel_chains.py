from langchain_groq import ChatGroq
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableParallel

load_dotenv()

# Initialize Groq model
model = ChatGroq(
    model="llama-3.3-70b-versatile",
    temperature=0.5
)

# Prompt for notes
prompt1 = PromptTemplate(
    template="Generate short and simple notes from the following text:\n\n{text}",
    input_variables=["text"]
)

# Prompt for quiz
prompt2 = PromptTemplate(
    template="Generate 5 short question-answer pairs from the following text:\n\n{text}",
    input_variables=["text"]
)

# Prompt for merging
prompt3 = PromptTemplate(
    template="""
Merge the provided notes and quiz into a single well-formatted document.

Notes:
{notes}

Quiz:
{quiz}
""",
    input_variables=["notes", "quiz"]
)

parser = StrOutputParser()

# Run notes and quiz generation in parallel
parallel_chain = RunnableParallel({
    "notes": prompt1 | model | parser,
    "quiz": prompt2 | model | parser
})

# Merge the outputs
merge_chain = prompt3 | model | parser

# Complete chain
chain = parallel_chain | merge_chain

# Input text
text = """
Support vector machines (SVMs) are a set of supervised learning methods used for classification, regression and outliers detection.

The advantages of support vector machines are:

Effective in high dimensional spaces.

Still effective in cases where number of dimensions is greater than the number of samples.

Uses a subset of training points in the decision function (called support vectors), so it is also memory efficient.

Versatile: different Kernel functions can be specified for the decision function. Common kernels are provided, but it is also possible to specify custom kernels.

The disadvantages of support vector machines include:

If the number of features is much greater than the number of samples, avoid over-fitting in choosing Kernel functions and regularization term is crucial.

SVMs do not directly provide probability estimates; these are calculated using an expensive five-fold cross-validation.

The support vector machines in scikit-learn support both dense and sparse sample vectors as input. For optimal performance, use C-ordered numpy.ndarray (dense) or scipy.sparse.csr_matrix (sparse) with dtype=float64.
"""

# Invoke chain
result = chain.invoke({"text": text})

print(result)

# Print chain graph
chain.get_graph().print_ascii()