import os
import streamlit as st

from dotenv import load_dotenv

from langchain_google_genai import (
    ChatGoogleGenerativeAI,
    GoogleGenerativeAIEmbeddings
)

from langchain_community.vectorstores import FAISS

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser


# --------------------------------
# 1. PAGE CONFIGURATION
# --------------------------------

st.set_page_config(
    page_title="College Information Assistant",
    page_icon="🎓",
    layout="wide"
)


# --------------------------------
# 2. LOAD ENVIRONMENT VARIABLES
# --------------------------------

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

load_dotenv(os.path.join(BASE_DIR, ".env"))

api_key = os.getenv("GOOGLE_API_KEY")

if not api_key:
    st.error("Google API key not found. Check your .env file.")
    st.stop()


# --------------------------------
# 3. INITIALIZE EMBEDDINGS
# --------------------------------

@st.cache_resource
def load_rag_components():

    embeddings = GoogleGenerativeAIEmbeddings(
        model="models/gemini-embedding-001",
        google_api_key=api_key
    )


    # --------------------------------
    # 4. LOAD FAISS VECTOR DATABASE
    # --------------------------------

    vector_db_path = os.path.join(BASE_DIR, "faiss_index")

    if not os.path.exists(
        os.path.join(vector_db_path, "index.faiss")
    ):
        raise FileNotFoundError(
            "FAISS index not found. Run load_documents.py first."
        )

    vectorstore = FAISS.load_local(
        vector_db_path,
        embeddings,
        allow_dangerous_deserialization=True
    )


    # --------------------------------
    # 5. CREATE RETRIEVER
    # --------------------------------

    retriever = vectorstore.as_retriever(
        search_kwargs={"k": 3}
    )


    # --------------------------------
    # 6. INITIALIZE GEMINI LLM
    # --------------------------------

    llm = ChatGoogleGenerativeAI(
        model="gemini-3.5-flash",
        temperature=0,
        google_api_key=api_key
    )


    # --------------------------------
    # 7. CREATE PROMPT TEMPLATE
    # --------------------------------

    prompt = ChatPromptTemplate.from_template("""

    You are a helpful College Information Assistant.

    Answer the student's question using ONLY the
    information available in the provided context.

    Instructions:

    1. Give simple and clear answers.
    2. Use bullet points when necessary.
    3. If the answer is not available in the context,
       say: "Sorry, this information is not available
       in the college documents."
    4. Do not invent information.
    5. Be polite and professional.

    Context:
    {context}

    Student Question:
    {question}

    Answer:

    """)


    # --------------------------------
    # 8. FORMAT RETRIEVED DOCUMENTS
    # --------------------------------

    def format_docs(docs):

        return "\n\n".join(
            doc.page_content for doc in docs
        )


    # --------------------------------
    # 9. CREATE RAG CHAIN
    # --------------------------------

    rag_chain = (

        {
            "context": retriever | format_docs,
            "question": RunnablePassthrough()
        }

        | prompt

        | llm

        | StrOutputParser()

    )

    return rag_chain, retriever


# --------------------------------
# 10. LOAD RAG COMPONENTS
# --------------------------------

try:

    rag_chain, retriever = load_rag_components()

except Exception as e:

    st.error(f"Error loading RAG system: {e}")
    st.stop()


# --------------------------------
# 11. SIDEBAR
# --------------------------------

with st.sidebar:

    st.title("🎓 College Assistant")

    st.write(
        "Ask questions about college admissions, "
        "courses, fees, faculty and facilities."
    )

    st.divider()

    st.subheader("Available Features")

    st.write("📚 Admission Information")
    st.write("💰 Fee Structure")
    st.write("🎓 Courses")
    st.write("👨‍🏫 Faculty Information")
    st.write("🏫 College Details")

    st.divider()

    if st.button("Clear Chat"):

        st.session_state.messages = []

        st.rerun()


# --------------------------------
# 12. MAIN INTERFACE
# --------------------------------

st.title("🎓 College Information Assistant")

st.caption(
    "AI-powered College Chatbot using "
    "Retrieval-Augmented Generation (RAG)"
)

st.divider()


# --------------------------------
# 13. INITIALIZE CHAT HISTORY
# --------------------------------

if "messages" not in st.session_state:

    st.session_state.messages = []


# --------------------------------
# 14. DISPLAY CHAT HISTORY
# --------------------------------

for message in st.session_state.messages:

    with st.chat_message(message["role"]):

        st.markdown(message["content"])


# --------------------------------
# 15. USER INPUT
# --------------------------------

question = st.chat_input(
    "Ask anything about your college..."
)


if question:

    # Display user question

    with st.chat_message("user"):

        st.markdown(question)

    st.session_state.messages.append({

        "role": "user",
        "content": question

    })


    # --------------------------------
    # 16. RETRIEVE DOCUMENTS
    # --------------------------------

    with st.chat_message("assistant"):

        with st.spinner("Searching college documents..."):

            try:

                # Retrieve relevant documents

                retrieved_docs = retriever.invoke(question)


                # --------------------------------
                # 17. GENERATE ANSWER
                # --------------------------------

                response = rag_chain.invoke(question)

                st.markdown(response)


                # --------------------------------
                # 18. DISPLAY SOURCE DOCUMENTS
                # --------------------------------

                with st.expander("📄 View Source Documents"):

                    if retrieved_docs:

                        for i, doc in enumerate(
                            retrieved_docs, start=1
                        ):

                            source = doc.metadata.get(
                                "source", "Unknown"
                            )

                            page = doc.metadata.get(
                                "page", "Unknown"
                            )

                            st.write(f"**Source {i}:** {os.path.basename(source)}")

                            st.write(f"Page: {page}")

                            st.write(doc.page_content[:300])

                            st.divider()

                    else:

                        st.write("No relevant documents found.")


                # Save assistant response

                st.session_state.messages.append({

                    "role": "assistant",
                    "content": response

                })


            except Exception as e:

                st.error(f"Something went wrong: {e}")