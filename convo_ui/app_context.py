from retrieval.vectorstore import VectorStoreManager
from data_ingestion.clip_embedder import load_clip_model
from langchain.embeddings import HuggingFaceEmbeddings
from langchain_openai import ChatOpenAI

llm = ChatOpenAI(model="gpt-4o-mini")

embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2") # use this in vectorstore

clip_embedding_model, clip_preprocess = load_clip_model(device="cpu")

vectorstore_manager = VectorStoreManager(embedding_model)