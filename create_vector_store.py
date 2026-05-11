import os
from dotenv import load_dotenv
from langchain_community.document_loaders.csv_loader import CSVLoader
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
# Load environment variables (ensure HF_TOKEN is in your .env if using a gated model)
load_dotenv()

def build_vector_store():
    csv_path = "AI_Resume_Screening.csv"
    persist_dir = "./candidate_vector_db"

    if not os.path.exists(csv_path):
        print(f"Error: Could not find {csv_path}. Please ensure the file exists.")
        return

    print("Initializing embedding model...")
    
    # EMBEDDING TOKEN PARAMETER:
    # Passed via environment variable or directly if required for gated models.
    hf_token = os.getenv("HF_TOKEN")
    
    embedding_model = HuggingFaceEmbeddings(
        model_name="BAAI/bge-small-en-v1.5",
        # Pass the token parameter to model_kwargs if your model requires authentication
        model_kwargs={'device': 'cpu', 'token': hf_token} 
    )

    print(f"Loading data from {csv_path}...")
    loader = CSVLoader(file_path=csv_path, encoding="utf-8")
    documents = loader.load()

    print("Creating and persisting Chroma vector store...")
    # This creates the DB and saves it to persist_dir
    vector_store = Chroma.from_documents(
        documents=documents, 
        embedding=embedding_model, 
        persist_directory=persist_dir
    )
    
    print(f"✅ Vector store successfully created and persisted at '{persist_dir}'")

if __name__ == "__main__":
    build_vector_store()