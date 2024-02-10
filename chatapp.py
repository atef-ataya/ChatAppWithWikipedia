import streamlit as st
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import Chroma
import os

# function that answers to user's query from Wikipedia
def load_wikipedia(query, lang='en', load_max_docs=10):
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
def chat_with_wikipedia(vector_store, query, k=3):
    from langchain.chains import RetrievalQA
    from langchain_openai import ChatOpenAI

    llm = ChatOpenAI(model='gpt-4', temperature=1)

    retriever = vector_store.as_retriever(search_type='similarity', search_kwargs={'k': k})

    chain = RetrievalQA.from_chain_type(llm=llm, chain_type="stuff", retriever=retriever)

    answer = chain.invoke(query)
    return answer

# Function to clear the history of chat from the session state
def clear_history():
    if 'history' in st.session_state:
        del st.session_state['history']

# Add application entry point
if __name__ == '__main__':
    import os
    from dotenv import load_dotenv, find_dotenv
    load_dotenv(find_dotenv(), override=True)

    # Top main page content
    st.image('./images/banner.jpg')
    st.subheader('Chat with Wikipedia')
    st.write(
        "Check out this repository on my github profile [link](https://github.com/atef-ataya/ChatAppWithWikipedia)")
    # creating the sidebar panel
    with st.sidebar:
        # saving the api key entered by the users into environment variable
        api_key = st.text_input('OpenAI API Key', type='password')
        subject = st.text_input('Please pick your topic from wikipedia')
        chunk_size = st.number_input('Chunk size:', min_value=100, max_value=2048, value=512, on_change=clear_history)
        k = st.number_input('K', min_value=1, max_value=20, value=3, on_change=clear_history)
        add_data = st.button('Load Data', on_click=clear_history)

        if subject and add_data and api_key:
            with st.spinner('Loading, Chunking and embedding file ...'):
                os.environ['OPENAI_API_KEY'] = api_key
                data = load_wikipedia(subject, lang='en')
                chunks = split_text_into_chunks(data)
                st.write(f'Chunk size: {chunk_size}, chunks: {len(chunks)}')
                tokens, embedding_cost = calculate_and_display_embedding_cost(chunks)
                st.write(f'Embedding cost: ${embedding_cost:.4f}')
                vector_store = create_embeddings(chunks)
                st.session_state.vs = vector_store
                st.success('Wikipedia information loaded, chunked, and embedded successfully!')
        else:
            st.write('Please provie your API_KEY and topic')

    q = st.text_input('Ask a question about the content of your file:')
    if q:
        if 'vs' in st.session_state:
            vector_store = st.session_state.vs
            with st.spinner('Getting the information ...'):
                answer = chat_with_wikipedia(vector_store, q, k)
                st.text_area('LLM Answer: ', value=answer['result'])

            st.divider()
            if 'history' not in st.session_state:
                st.session_state.history = ''
            value = f'Q: {q} \nA:{answer}'
            st.session_state.history = f'{value} \n {"-" * 100} \n {st.session_state.history}'
            h = st.session_state.history
            st.text_area('Chat history:', value=h, key='history', height=400)
