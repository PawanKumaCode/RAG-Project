from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_nvidia_ai_endpoints import ChatNVIDIA
from qdrant_client.http.models import Distance, VectorParams
from qdrant_client import QdrantClient
from langchain_qdrant import QdrantVectorStore
from langchain_community.embeddings.fastembed import FastEmbedEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from dotenv import load_dotenv
import os
import warnings

warnings.filterwarnings("ignore")


# Load API Key
load_dotenv()

api_key = os.getenv("NVIDIA_API_KEY")

# PDF Path
pdf_path = os.path.join(
    os.path.dirname(__file__),
    "document_loader",
    "STATS.pdf"
)

print(f"Loading PDF: {pdf_path}")

loader = PyPDFLoader(pdf_path)
docs = loader.load()

print(f"Loaded {len(docs)} pages.")

# Chunking
splitter = RecursiveCharacterTextSplitter(
    chunk_size=600,
    chunk_overlap=80
)

chunks = splitter.split_documents(docs)

print(f"Created {len(chunks)} chunks.")

# Embeddings
embeddings = FastEmbedEmbeddings()

client = QdrantClient(location=":memory:")

client.create_collection(
    collection_name="rag_collection",
    vectors_config=VectorParams(
        size=384,
        distance=Distance.COSINE
    )
)

vectorstore = QdrantVectorStore(
    client=client,
    collection_name="rag_collection",
    embedding=embeddings
)

vectorstore.add_documents(chunks)

print("Documents indexed successfully.")

# Retriever
retriever = vectorstore.as_retriever(
    search_kwargs={"k": 3}
)

# LLM
llm = ChatNVIDIA(
    model="nvidia/nemotron-3.5-lightning-30b-a3b",
    nvidia_api_key=api_key,
    temperature=0.2,
    max_completion_tokens=2500
)

# Prompt
prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """
You are an expert tutor.

Answer only from the provided context.

If answer is not available in context, say:
'I do not know based on the provided document.'

Context:
{context}
"""
        ),
        ("human", "{question}")
    ]
)

rag_chain = prompt | llm | StrOutputParser()

# Query Function


def query_rag(question):

    docs = retriever.invoke(question)

    context = "\n\n".join(
        [doc.page_content for doc in docs]
    )

    response = rag_chain.invoke(
        {
            "context": context,
            "question": question
        }
    )

    return response


if __name__ == "__main__":

    question = input("Ask Question: ")

    answer = query_rag(question)

    print("\nAnswer:")
    print(answer)
