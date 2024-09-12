import os

import streamlit as st
from src.helper import load_embedding_model,repo_ingestion
from dotenv import load_dotenv
from langchain.memory import ConversationSummaryMemory
from langchain.chains import ConversationalRetrievalChain
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain.vectorstores import Chroma


load_dotenv()

embedding = load_embedding_model()
persist_directory="data"

vectordb = Chroma(persist_directory=persist_directory,embedding_function=embedding)

llm = ChatOpenAI()
memory = ConversationSummaryMemory(llm=llm,memory_key="chat_history",return_messages=True)
chain = ConversationalRetrievalChain.from_llm(llm,
                                              retriever=vectordb.as_retriever(search_type="mmr",search_kwargs={"k":3}),
                                              memory=memory)


st.title("Source Code Analysis")



repo = st.text_input("Enter the repo url:")

if st.button("Ingest Repo"):
    repo_ingestion(repo)
    os.system("python store_index.py")

question = st.text_input("Enter the question:")

if question == "clear":
    os.system("rmdir /S /Q repo")
elif question != "":
    result = chain.invoke(question)
    st.write(result["answer"])

