# Streamlit App Code
import streamlit as st
from urllib.parse import urljoin, urlparse
from collections import deque
import requests
from bs4 import BeautifulSoup
from llama_index.readers.web import SimpleWebPageReader
from llama_index.core import Settings, VectorStoreIndex, StorageContext
from llama_index.core.memory import ChatMemoryBuffer
from llama_index.vector_stores.qdrant import QdrantVectorStore
from llama_index.llms.openai import OpenAI
from llama_index.embeddings.fastembed import FastEmbedEmbedding
import os

# Get API Keys
os.environ["OPENAI_API_KEY"] = st.secrets["OPENAI_API_KEY"]

# Set Streamlit page configuration
st.set_page_config(page_title="Website Chat Assistant", layout="centered")

# Application Overview
st.markdown("""
    **Overview:**  
    This application allows users to input a website URL. The app then crawls all links on the website 
    that share the same root domain as the provided URL. Once the links are gathered, the app uses OpenAI's 
    LLM with Llama-Index to create a retrieval-based chain for answering questions about the website's content.

    **Features:**  
    - Crawls and extracts links from the provided website, ensuring all links belong to the same root domain.  
    - Processes the content of the extracted links and indexes it for efficient querying.  
    - Utilizes OpenAI's GPT-powered Llama-Index for creating a chat interface that allows users to ask questions about the website.  
    - Integrates embeddings for semantic search and retrieval capabilities.
""")

# Title
st.title("Website Chat Assistant")

# Initialize session state
if "chat_ready" not in st.session_state:
    st.session_state.chat_ready = False
if "chat_engine" not in st.session_state:
    st.session_state.chat_engine = None

# Function for extracting links
def extract_links(base_url):
    visited = set()
    to_visit = deque([base_url])
    all_links = set()

    while to_visit:
        current_url = to_visit.popleft()
        if current_url in visited:
            continue

        visited.add(current_url)

        try:
            response = requests.get(current_url)
            if response.status_code != 200:
                continue
            
            soup = BeautifulSoup(response.text, 'html.parser')

            for a_tag in soup.find_all('a', href=True):
                href = a_tag['href']
                absolute_url = urljoin(current_url, href)

                if urlparse(absolute_url).netloc == urlparse(base_url).netloc:
                    if absolute_url not in visited:
                        to_visit.append(absolute_url)
                        all_links.add(absolute_url)

        except requests.exceptions.RequestException as e:
            st.error(f"Error fetching {current_url}: {e}")

    return list(all_links)

# Input for website URL
base_url = st.text_input("Enter Website URL:", placeholder="https://example.com")

if st.button("Process Website") and base_url:
    with st.spinner("Extracting links and setting up..."):
        links = extract_links(base_url)
        if not links:
            st.error("No links extracted. Please try a different website.")
        else:
            st.success(f"Extracted {len(links)} links from the website.")
            
            # LLM and embedding setup
            llm = OpenAI()
            embedding_model = FastEmbedEmbedding()
            Settings.llm = llm
            Settings.embed_model = embedding_model

            # Load and index documents
            documents = SimpleWebPageReader(html_to_text=True).load_data(links)
            storage_context = StorageContext.from_defaults()
            index = VectorStoreIndex.from_documents(
                documents, storage_context=storage_context
            )
            query_engine = index.as_query_engine()
            memory = ChatMemoryBuffer.from_defaults(token_limit=3000)
            st.session_state.chat_engine = index.as_chat_engine(
                chat_mode="context",
                memory=memory,
                system_prompt="You are an AI assistant who answers the user's questions."
            )
            st.session_state.chat_ready = True

# Chat Interface
if st.session_state.chat_ready:
    st.subheader("Ask Questions About the Website")
    user_query = st.text_input("Enter your question:")
    if st.button("Chat") and user_query:
        response = st.session_state.chat_engine.chat(user_query)
        st.write("**AI Assistant:**", response.response)
