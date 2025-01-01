import streamlit as st
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, pipeline
from PyPDF2 import PdfReader
from langchain.schema import Document
import os
from transformers.utils import default_cache_path
import re
import spacy
from spacy.util import is_package

llm_model_names = ["google/flan-t5-small", "google/flan-t5-large"]
existing_llm_models = []

embedding_model_names = ["sentence-transformers/all-MiniLM-L6-v2"]
existing_embeding_models = []

def find_models(cache_path=default_cache_path):
    folder_names = os.listdir(cache_path)
    
    for model in llm_model_names:
        model = model.replace("/", "--")
        for folder_name in folder_names:
            if re.search(model, folder_name):
                existing_llm_models.append(model)
     
    for model in embedding_model_names:
        model = model.replace("/", "--")
        for folder_name in folder_names:
            if re.search(model, folder_name):
                existing_embeding_models.append(model)

def download_llm_model(model_name):
    AutoTokenizer.from_pretrained(model_name)
    AutoModelForSeq2SeqLM.from_pretrained(model_name)

def download_embedding_model(model_name):
    HuggingFaceEmbeddings(model_name=model_name)

# Preprocessing: Lemmatization
def preprocess_text_with_lemmatization(text):
    
    model_name = "en_core_web_sm"
    
    if not is_package(model_name):
        print(f"{model_name} not found. Downloading...")
        spacy.cli.download(model_name)
    else:
        print(f"{model_name} is already installed.")
        
    # Load the model
    nlp = spacy.load(model_name)
    
    doc = nlp(text)
    lemmatized_text = " ".join([token.lemma_ for token in doc])
    return lemmatized_text

# Step 3: Load Hugging Face model once
@st.cache_resource(show_spinner=False)
def load_model(model_name):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
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

# Step 4: Load and preprocess PDF
@st.cache_data(show_spinner=False)
def load_and_preprocess_pdf(pdf_path):
    reader = PdfReader(pdf_path)
    text = ""
    for page in reader.pages:
        text += page.extract_text()
    # Apply lemmatization to the extracted text
    lemmatized_text = preprocess_text_with_lemmatization(text)
    return lemmatized_text

# Step 5: Split text into manageable chunks
@st.cache_data(show_spinner=False)
def chunk_text(text):
    splitter = RecursiveCharacterTextSplitter(chunk_size=2000, chunk_overlap=400)
    return splitter.create_documents([text])

# Step 6: Embed and index chunks
@st.cache_data(show_spinner=False)
def create_vector_store(_docs, embedding_model_name):
    documents = [Document(page_content=doc.page_content) for doc in _docs]
    embeddings = HuggingFaceEmbeddings(model_name=embedding_model_name)
    vector_store = FAISS.from_documents(documents, embeddings)
    return vector_store

# Step 7: Answer questions using a free LLM
def answer_question(vector_store, model_name, question):
    docs = vector_store.similarity_search(question, k=5)
    
    # Combine chunks while respecting token limit
    context = ""
    max_token_length = 512
    for doc in docs:
        if len(context.split()) + len(doc.page_content.split()) <= max_token_length:
            context += " " + doc.page_content
    
    # Query
    refined_query = f"""Using the following context, provide a concise and specific answer to the question: "{question}"
    Context:
    {context}
    """
    
    hf_pipeline = load_model(model_name) 
    result = hf_pipeline(refined_query)
    answer = result[0]['generated_text']
    return answer, docs

# Streamlit app   
def main():
        
    model_availability = {"LLM": False, "Embedding": False}
    find_models()
    
    # Set up sidebar
    st.sidebar.title("Model Selection")
    
    # Choose LLM model
    selected_llm_model_name = st.sidebar.selectbox("Choose or search the LLM model", llm_model_names)
    formated_llm_model_name = selected_llm_model_name.replace("/", "--")
    
    if formated_llm_model_name in existing_llm_models:
        st.sidebar.success(f"Model '{selected_llm_model_name}' is already available.")
        model_availability["LLM"] = True
    else:
        st.sidebar.warning(f"Model '{selected_llm_model_name}' is not available.")    
        if st.sidebar.button("Download LLM Model"):
            with st.spinner(f"Downloading {selected_llm_model_name} ..."):
                download_llm_model(selected_llm_model_name)
                model_availability["LLM"] = True
            st.experimental_rerun()
    
    # Choose embedding model 
    selected_embedding_model_name = st.sidebar.selectbox("Choose or search the embedding model", embedding_model_names)
    formated_embedding_model_name = selected_embedding_model_name.replace("/", "--")
    
    if formated_embedding_model_name in existing_embeding_models:
        st.sidebar.success(f"Model '{selected_embedding_model_name}' is already available.")
        model_availability["Embedding"] = True
    else:
        st.sidebar.warning(f"Model '{selected_embedding_model_name}' is not available.")
        if st.sidebar.button("Download Embedding Model"):
            with st.spinner(f"Downloading {selected_embedding_model_name} ..."):
                download_embedding_model(selected_embedding_model_name)
                model_availability["Embedding"] = True
            st.experimental_rerun()
                   
    # Main body
    st.title("PDF Question-Answering App")
    
    if "pdf_name" not in st.session_state:
        st.session_state.pdf_name = None
    
    if model_availability["LLM"] and model_availability["Embedding"]:
        # File uploader
        uploaded_file = st.file_uploader("Upload a PDF file", type="pdf")
        
        if uploaded_file is not None:
            
            # Clear cache if a new PDF is uploaded
            if st.session_state.pdf_name != uploaded_file.name:
                
                st.session_state.pdf_name = uploaded_file.name
                st.cache_data.clear()
                st.cache_resource.clear()
            
            with st.spinner("Processing PDF..."):
                text = load_and_preprocess_pdf(uploaded_file)
                docs = chunk_text(text)
                vector_store = create_vector_store(docs, selected_embedding_model_name)
                st.success("PDF processed successfully! You can now ask questions.")
            
            # User question input  
            question = st.text_input("Ask a question about the PDF content:")

            if question:
                with st.spinner("Searching for the answer..."):
                    answer, sources = answer_question(vector_store, selected_llm_model_name, question)
                
                # Display the answer
                st.subheader("Answer")
                st.write(answer)
                
                # Display sources
                if sources:
                    st.subheader("Relevant Sources")
                    for i, doc in enumerate(sources):
                        st.write(f"**Source {i + 1}:**")
                        st.write(f"{doc.page_content[:1000]}...")
    else:
        st.warning("The required models are not available. Please download them to proceed.")

# Run the Streamlit app
if __name__ == "__main__":
    main()