import streamlit as st
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import Chroma
import os

# function that answers to user's query from Wikipedia
def chat_with_wikipedia(query, lang='en', load_max_docs=2):
    from langchain.document_loaders import WikipediaLoader
    loader = WikipediaLoader(query=query, lang=lang, load_max_docs=load_max_docs)
    data = loader.load()
    return data

# Function for chunking text
def split_text_into_chunks(data, chunk_size=256, chunk_overlap=20):
    from langchain.text_splitter import RecursiveCharacterTextSplitter
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    chunks = text_splitter.split_documents(data)
    return chunks

# Function to create the embedding from chunks and return the vector store
def create_embeddings(chunks):
    embeddings = OpenAIEmbeddings()
    vector_store = Chroma.from_documents(chunks, embeddings)
    return vector_store

# Function to calculate the cost of embeddings
def calculate_and_display_embedding_cost(texts):
    import tiktoken
    enc = tiktoken.encoding_for_model('text-embedding-ada-002')
    total_tokens = sum([len(enc.encode(page.page_content)) for page in texts])
    return total_tokens, total_tokens / 1000 * 0.0004

# Function that allows us to chat with Wikipedia
def chat_app_with_wikipedia(vector_store, query, chat_history=[], k=3):
    from langchain.chains import ConversationalRetrievalChain
    from langchain_openai import ChatOpenAI

    llm = ChatOpenAI(temperature=1)
    retriever = vector_store.as_retriever(search_type='similarity', search_kwargs={'k': k})

    crc = ConversationalRetrievalChain.from_llm(llm, retriever)
    result = crc.invoke({'question': query, 'chat_history': chat_history})
    chat_history.append((query, result['answer']))

    return result, chat_history

# Function to clear the history of chat from the session state
def clear_history():
    if 'history' in st.session_state:
        del st.session_state['history']

# Add application entry point
