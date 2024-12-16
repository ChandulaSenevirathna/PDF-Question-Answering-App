import streamlit as st
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, pipeline
from PyPDF2 import PdfReader
from langchain.schema import Document

# Step 1: Load Hugging Face model once
@st.cache_resource(show_spinner=False)
def load_model():
    tokenizer = AutoTokenizer.from_pretrained("google/flan-t5-large")
    model = AutoModelForSeq2SeqLM.from_pretrained("google/flan-t5-large")
    hf_pipeline = pipeline(
        "text2text-generation",
        model=model,
        tokenizer=tokenizer,
        truncation=True,
        max_length=512,
        num_beams=3,  
        do_sample=False
    )
    return hf_pipeline

hf_pipeline = load_model()

# Step 2: Load PDF and extract text
@st.cache_data(show_spinner=False)  # Suppress cache spinner
def load_pdf(pdf_path):
    reader = PdfReader(pdf_path)
    text = ""
    for page in reader.pages:
        text += page.extract_text()
    return text

# Step 3: Split text into manageable chunks
@st.cache_data(show_spinner=False)  
def chunk_text(text):
    splitter = RecursiveCharacterTextSplitter(chunk_size=2000, chunk_overlap=400)
    return splitter.create_documents([text])

# Step 4: Embed and index chunks
@st.cache_data(show_spinner=False) 
def create_vector_store(_docs):
    documents = [Document(page_content=doc.page_content) for doc in _docs]
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    vector_store = FAISS.from_documents(documents, embeddings)
    return vector_store

# Step 5: Answer questions using a free LLM
def answer_question(vector_store, question):

    docs = vector_store.similarity_search(question, k=5)
    
    # Combine chunks while respecting token limit
    context = ""
    max_token_length = 512
    for doc in docs:
        if len(context.split()) + len(doc.page_content.split()) <= max_token_length:
            context += " " + doc.page_content
    
    # query
    refined_query = f"""Using the following context, provide a concise and specific answer to the question: "{question}"
    Context:
    {context}
    """
    
    result = hf_pipeline(refined_query)
    answer = result[0]['generated_text']
    return answer, docs

# Streamlit App
st.title("PDF Question-Answering App")

# File uploader
uploaded_file = st.file_uploader("Upload a PDF file", type="pdf")

if uploaded_file is not None:
    with st.spinner("Processing PDF..."):
        
        text = load_pdf(uploaded_file)
        docs = chunk_text(text)
        vector_store = create_vector_store(docs)
        st.success("PDF processed successfully! You can now ask questions.")
    
    # user question input
    question = st.text_input("Ask a question about the PDF content:")
    
    if question:
        with st.spinner("Searching for the answer..."):
            answer, sources = answer_question(vector_store, question)
        
        # Display the answer
        st.subheader("Answer")
        st.write(answer)
        
        # Display sources
        if sources:
            st.subheader("Relevant Sources")
            for i, doc in enumerate(sources):
                st.write(f"**Source {i + 1}:**")
                st.write(f"{doc.page_content[:1000]}...")
