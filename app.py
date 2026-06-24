import os
import streamlit as st

from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_groq import ChatGroq

# --------------------------------------------------

# CONFIG

# --------------------------------------------------

LLM_MODEL = "llama-3.3-70b-versatile"
CORPUS_PATH = "."

# --------------------------------------------------

# STREAMLIT PAGE

# --------------------------------------------------

st.set_page_config(
page_title="Zyro Dynamics HR Assistant",
page_icon="🤖"
)

st.title("🤖 Zyro Dynamics HR Help Desk")

st.markdown(
"Ask questions about leave policy, compensation, benefits, work from home, onboarding, POSH and other HR policies."
)

# --------------------------------------------------

# LOAD RAG COMPONENTS

# --------------------------------------------------

@st.cache_resource
def load_rag():

```
loader = PyPDFDirectoryLoader(CORPUS_PATH)
documents = loader.load()

splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=100
)

chunks = splitter.split_documents(documents)

embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

vectorstore = FAISS.from_documents(
    chunks,
    embeddings
)

retriever = vectorstore.as_retriever(
    search_type="mmr",
    search_kwargs={
        "k": 5,
        "fetch_k": 20
    }
)

llm = ChatGroq(
    model=LLM_MODEL,
    temperature=0.1,
    max_tokens=512
)

return retriever, llm
```

retriever, llm = load_rag()

# --------------------------------------------------

# PROMPTS

# --------------------------------------------------

RAG_PROMPT = ChatPromptTemplate.from_template(
"""
You are the Zyro Dynamics HR Help Desk Assistant.

Answer the employee's question ONLY using the provided HR policy context.

If the answer is not present in the context, respond:

"I could not find that information in the Zyro Dynamics HR policy documents."

Context:
{context}

Question:
{question}

Answer:
"""
)

OOS_PROMPT = ChatPromptTemplate.from_template(
"""
You are a classifier.

Determine whether the question is related to Zyro Dynamics HR policies.

Topics include:

* leave policy
* payroll
* benefits
* compensation
* onboarding
* separation
* work from home
* code of conduct
* POSH
* performance reviews
* travel reimbursement
* employee handbook
* HR policies

Question:
{question}

Respond with ONLY:

YES

or

NO
"""
)

REFUSAL_MESSAGE = (
"I can only answer questions related to Zyro Dynamics HR policy documents."
)

# --------------------------------------------------

# HELPERS

# --------------------------------------------------

def format_docs(docs):
return "\n\n".join(doc.page_content for doc in docs)

def rag_chain(question):

```
chain = (
    {
        "context": retriever | format_docs,
        "question": RunnablePassthrough()
    }
    | RAG_PROMPT
    | llm
    | StrOutputParser()
)

return chain.invoke(question)
```

def ask_bot(question):

```
classifier_chain = (
    OOS_PROMPT
    | llm
    | StrOutputParser()
)

result = classifier_chain.invoke(
    {"question": question}
).strip().upper()

if result == "NO":
    return {
        "answer": REFUSAL_MESSAGE,
        "sources": []
    }

docs = retriever.invoke(question)

answer = rag_chain(question)

sources = list(
    set(
        [
            os.path.basename(
                doc.metadata.get("source", "")
            )
            for doc in docs
        ]
    )
)

return {
    "answer": answer,
    "sources": sources
}
```

# --------------------------------------------------

# CHAT UI

# --------------------------------------------------

question = st.chat_input(
"Ask an HR question..."
)

if question:

```
with st.chat_message("user"):
    st.markdown(question)

result = ask_bot(question)

with st.chat_message("assistant"):

    st.markdown(result["answer"])

    if result["sources"]:

        st.markdown("### Sources")

        for src in result["sources"]:
            st.markdown(f"- {src}")
```
