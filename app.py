import streamlit as st
import pickle
import faiss
import ollama
from sentence_transformers import SentenceTransformer

st.set_page_config(
    page_title="Document Q&A System",
    page_icon="📄",
    layout="centered"
)

st.title("📄 Document Q&A System")
st.write("Ask questions about the uploaded document.")

@st.cache_resource
def load_components():

    with open("vector_db/document_chunks.pkl", "rb") as f:
        document_chunks = pickle.load(f)

    document_index = faiss.read_index(
        "vector_db/document_faiss_index.bin"
    )

    embedding_model = SentenceTransformer(
        "all-MiniLM-L6-v2"
    )

    return document_chunks, document_index, embedding_model

document_chunks, document_index, embedding_model = load_components()

def ask_rag(question):

    query_embedding = embedding_model.encode([question])

    distances, indices = document_index.search(
        query_embedding.astype("float32"),
        k=1
    )

    retrieved_document = document_chunks[indices[0][0]]

    context = retrieved_document.page_content

    source = retrieved_document.metadata["source"]

    page = retrieved_document.metadata["page_label"]


    prompt = f"""
You are a document question-answering assistant.

Answer the question using ONLY the information provided in the context.

If the answer is not present in the context, say:
"I could not find the answer in the provided document."

Context:
{context}

Question:
{question}

Answer:
"""


    
    response = ollama.chat(
        model="llama3.2",
        messages=[
            {
                "role": "user",
                "content": prompt
            }
        ]
    )


    answer = response["message"]["content"]

    return answer, source, page




question = st.text_input(
    "Enter your question:"
)

if st.button("Ask"):

    if question.strip():

        with st.spinner("Searching document..."):

            answer, source, page = ask_rag(question)


        st.subheader("Answer")

        st.write(answer)


        st.subheader("Source")

        st.write(source)


        st.write(f"Page: {page}")

    else:

        st.warning("Please enter a question.")