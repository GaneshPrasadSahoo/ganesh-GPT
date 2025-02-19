import streamlit as st
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
import os
from langchain_google_genai import GoogleGenerativeAIEmbeddings
import google.generativeai as genai
from langchain.vectorstores import FAISS
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate
from dotenv import load_dotenv

# Load environment variables
load_dotenv()
api_key = os.getenv("GOOGLE_API_KEY")

# Configure Google Generative AI with API key
genai.configure(api_key=api_key)

# Function to extract text from PDF
def get_pdf_text(pdf_docs):
    text = ""
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            text += page.extract_text()
    return text

# Function to split text into chunks
def get_text_chunks(text):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=10000, chunk_overlap=1000)
    chunks = text_splitter.split_text(text)
    return chunks

# Function to generate vector store from text chunks
def get_vector_store(text_chunks):
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    vector_store = FAISS.from_texts(text_chunks, embedding=embeddings)
    vector_store.save_local("faiss_index")

# Function to load the conversational chain for PDF Q&A
def get_conversational_chain():
    prompt_template = """
    Answer the question as detailed as possible from the provided context. If the answer is not in
    the provided context, just say "answer is not available in the context" and don't provide a wrong answer.

    Context:
    {context}

    Question: 
    {question}

    Answer:
    """
    model = ChatGoogleGenerativeAI(model="gemini-pro", temperature=0.3)
    prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
    chain = load_qa_chain(model, chain_type="stuff", prompt=prompt)
    return chain

# Function to process user question from PDFs
def chat_with_pdf(user_question):
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    
    # Load FAISS vector store safely
    new_db = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)
    
    docs = new_db.similarity_search(user_question)
    chain = get_conversational_chain()
    
    response = chain({"input_documents": docs, "question": user_question}, return_only_outputs=True)
    
    st.write("📄 **PDF Response:** ", response["output_text"])

# Function to chat with AI (general queries)
def chat_with_ai(user_query):
    model = ChatGoogleGenerativeAI(model="gemini-pro", temperature=0.7)
    response = model.invoke(user_query)
    
    # Extract and display AI-generated content
    st.write("🤖 **AI Response:** ", response.content)

# Main Streamlit application
def main():
    st.set_page_config(page_title="Chat with PDF & AI")
    st.header("GANESH GPT💁")

    # Tabs for different functionalities
    tab1, tab2 = st.tabs(["📄 Chat with PDF", "🤖 Chat with AI"])

    with tab1:
        st.subheader("Ask a Question from the PDF Files")
        user_question = st.text_input("Enter your question based on uploaded PDFs")
        if user_question:
            chat_with_pdf(user_question)

        with st.sidebar:
            st.title("Menu:")
            pdf_docs = st.file_uploader("Upload your PDF Files", accept_multiple_files=True)
            if st.button("Submit & Process"):
                with st.spinner("Processing..."):
                    raw_text = get_pdf_text(pdf_docs)
                    text_chunks = get_text_chunks(raw_text)
                    get_vector_store(text_chunks)
                    st.success("✅ PDFs processed successfully!")

    with tab2:
        st.subheader("Chat with AI 🤖")
        ai_question = st.text_input("Ask anything to AI")
        if ai_question:
            chat_with_ai(ai_question)

if __name__ == "__main__":
    main()
