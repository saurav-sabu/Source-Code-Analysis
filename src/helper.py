from git import Repo
import os

from langchain.text_splitter import Language, RecursiveCharacterTextSplitter
from langchain.document_loaders.generic import GenericLoader
from langchain.document_loaders.parsers import LanguageParser
from langchain_openai import OpenAIEmbeddings, ChatOpenAI

from langchain.memory import ConversationSummaryMemory
from langchain.chains import ConversationalRetrievalChain

def repo_ingestion(repo_url):
    os.makedirs("repo",exist_ok=True)
    repo_path = "repo/"
    Repo.clone_from(repo_url,to_path=repo_path)


def load_repo(repo_path):
    loader = GenericLoader.from_filesystem(repo_path,
                                       glob="**/*",
                                       suffixes=[".py"],
                                       parser=LanguageParser(language=Language.PYTHON,parser_threshold=500))
    
    document = loader.load()

    return document


def text_splitter(documents):
    splitter = RecursiveCharacterTextSplitter.from_language(language=Language.PYTHON,
                                                        chunk_size=2000,
                                                        chunk_overlap=200)
    
    texts = splitter.split_documents(documents)

    return texts

def load_embedding_model():
    embedding = OpenAIEmbeddings(disallowed_special=())
    return embedding
