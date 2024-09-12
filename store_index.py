from langchain.vectorstores import Chroma
import os
from src.helper import load_embedding_model, load_repo,text_splitter,repo_ingestion
from dotenv import load_dotenv

load_dotenv()

os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")

documents = load_repo("repo/")
text_chunks = text_splitter(documents)
embedding = load_embedding_model()

vectordb = Chroma.from_documents(text_chunks,embedding=embedding,persist_directory="./data")