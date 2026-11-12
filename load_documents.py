import os

from dotenv import load_dotenv
from langchain_community.document_loaders import DirectoryLoader, PyPDFLoader

# Import Text Splitter
from langchain_text_splitters import RecursiveCharacterTextSplitter

load_dotenv()
google_api_key = os.getenv("")
if not google_api_key:
    raise RuntimeError("Set GOOGLE_API_KEY in the environment or a .env file.")


# Step 1: Load PDF documents
loader = DirectoryLoader(
    r"C:\Users\ganga\Desktop\RAG_Project\documents",
    glob="*.pdf",
    loader_cls=PyPDFLoader
)

documents = loader.load()

print("Total Pages:", len(documents))


# Step 2: Create Text Splitter
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=100
)


# Step 3: Split Documents
chunks = text_splitter.split_documents(documents)


# Step 4: Check total chunks
print("Total Chunks:", len(chunks))


# Step 5: Display chunks
for i, chunk in enumerate(chunks):
    print(f"\n--- Chunk {i+1} ---")
    print(chunk.page_content)
    print(chunk.metadata)


from langchain_google_genai import GoogleGenerativeAIEmbeddings

# Create embedding model
embeddings = GoogleGenerativeAIEmbeddings(
    model="models/gemini-embedding-001",
    google_api_key=google_api_key
)

print("Gemini Embedding Model Created Successfully!")


# Step 6: Generate Embeddings for All Chunks
texts = [chunk.page_content for chunk in chunks]

vectors = embeddings.embed_documents(texts)


# Step 7: Display Embeddings
print("\nTotal Embeddings:", len(vectors))

for i, vector in enumerate(vectors):
    print(f"\n--- Embedding {i+1} ---")
    print("Vector Length:", len(vector))
    print("First 10 Values:", vector[:10])

# Import FAISS
from langchain_community.vectorstores import FAISS

# Create FAISS Vector Database
vectorstore = FAISS.from_documents(
    documents=chunks,
    embedding=embeddings
)

# Check total vectors
print("FAISS Database Created Successfully!")
print("Total Vectors:", vectorstore.index.ntotal)

# Save FAISS Database
vectorstore.save_local("faiss_index")

print("FAISS Database Saved Successfully!")   


query = "What is the MCA course fee?"

results = vectorstore.similarity_search(
    query,
    k=3
)
for i, doc in enumerate(results):
    print(f"\nResult {i+1}:")
    print(doc.page_content)
    print(doc.metadata)


# Step 1: Create Retriever
retriever = vectorstore.as_retriever(
    search_kwargs={"k": 3}
)

# Step 2: User Question
question = "What is the MCA course fee?"

# Step 3: Retrieve Documents
results = retriever.invoke(question)

# Step 4: Display Results
for i, doc in enumerate(results):
    print(f"\nResult {i+1}:")
    print(doc.page_content)
    print(doc.metadata)
    
from langchain_google_genai import ChatGoogleGenerativeAI

llm = ChatGoogleGenerativeAI(
    model="gemini-3.6-flash",
    temperature=0
)   
from langchain_core.prompts import ChatPromptTemplate

prompt = ChatPromptTemplate.from_template("""
You are a helpful College Information Assistant.

Answer the user's question using only the information
provided in the context.

If the answer is not available in the context,
say: I don't know based on the available information.

Context:
{context}

Question:
{question}

Answer:
""")
print(prompt.input_variables)  

from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

rag_chain = (
    {
        "context": retriever | format_docs,
        "question": RunnablePassthrough()
    }
    | prompt
    | llm
    | StrOutputParser()
)
question = "What is the MCA course fee?"

response = rag_chain.invoke(question)

print("Answer:", response)
