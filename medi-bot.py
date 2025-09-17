import os
import streamlit as st
from dotenv import load_dotenv

from langchain_huggingface import HuggingFaceEmbeddings
from langchain.chains import RetrievalQA
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import PromptTemplate
from langchain_groq import ChatGroq

# Load environment variables

load_dotenv()  # Load .env file
GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")

if not GROQ_API_KEY:
    st.error("GROQ_API_KEY is missing!")
    st.stop()

# Path to your FAISS database
DB_FAISS_PATH = "vectorstore/db_faiss"

# Cache vectorstore to avoid reloading

@st.cache_resource
def get_vectorstore():
    try:
        embedding_model = HuggingFaceEmbeddings(model_name='sentence-transformers/all-MiniLM-L6-v2')
        db = FAISS.load_local(DB_FAISS_PATH, embedding_model, allow_dangerous_deserialization=True)
        return db
    except Exception as e:
        st.error(f"Failed to load FAISS vector store: {e}")
        return None


# Custom Prompt Template

CUSTOM_PROMPT_TEMPLATE = """
You are a helpful medical assistant. Use the provided context to answer the question accurately.
- If the context doesn't contain enough information, say: "Based on the available information, here is what I found..."
- NEVER make up answers or provide unverified information.

Context:
{context}

Question:
{question}

Answer:
"""

def set_custom_prompt(template_text):
    return PromptTemplate(template=template_text, input_variables=["context", "question"])


# Streamlit UI

def main():
    st.set_page_config(page_title="MediBot")
    st.title(" MediBot ")
    st.markdown(
        "How can i help you?"
    )

    # Initialize chat history
    if 'messages' not in st.session_state:
        st.session_state.messages = []

    # Display chat history
    for message in st.session_state.messages:
        st.chat_message(message['role']).markdown(message['content'])

    # User input
    user_input = st.chat_input("Type your medical question here...")

    if user_input:
        st.chat_message('user').markdown(user_input)
        st.session_state.messages.append({'role': 'user', 'content': user_input})

        # Load FAISS database
        vectorstore = get_vectorstore()
        if vectorstore is None:
            return

        try:
           
            # Groq LLM Integration
           
            llm = ChatGroq(
                model_name="meta-llama/llama-4-maverick-17b-128e-instruct",  # Use a high-quality Groq model
                temperature=0.2,
                groq_api_key=GROQ_API_KEY
            )

            # Setup Retrieval QA Chain
            qa_chain = RetrievalQA.from_chain_type(
                llm=llm,
                chain_type="stuff",
                retriever=vectorstore.as_retriever(search_kwargs={'k': 3}),
                return_source_documents=True,
                chain_type_kwargs={'prompt': set_custom_prompt(CUSTOM_PROMPT_TEMPLATE)}
            )

            with st.spinner("Thinking... "):
                response = qa_chain.invoke({'query': user_input})

            result = response.get("result", "I couldn't find a suitable answer.")
            sources = "\n".join(
                [f"- {doc.metadata.get('source', 'Unknown')}" for doc in response.get("source_documents", [])]
            )

            
            final_response = f"**Answer:**\n{result}\n\n**Source Documents:**\n{sources if sources else 'No sources available.'}"

            st.chat_message('assistant').markdown(final_response)
            st.session_state.messages.append({'role': 'assistant', 'content': final_response})

        except Exception as e:
            st.error(f" Error during Groq response generation: {str(e)}")

# Run the Streamlit app
if __name__ == "__main__":
    main()
