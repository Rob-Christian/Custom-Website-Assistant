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
from llama_index.llms.gemini import Gemini
from llama_index.embeddings.fastembed import FastEmbedEmbedding
import qdrant_client
import os

# Get API Keys
os.environ["GOOGLE_API_KEY"] = st.secrets("GOOGLE_API_KEY")

# Set Streamlit page configuration
st.set_page_config(page_title="Website Chat Assistant", layout="centered")

# Title
st.title("Website Chat Assistant")

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
chat_ready = False

if st.button("Process Website") and base_url:
    with st.spinner("Extracting links and setting up..."):
        links = extract_links(base_url)
        if not links:
            st.error("No links extracted. Please try a different website.")
        else:
            st.success(f"Extracted {len(links)} links from the website.")
            
            # LLM and embedding setup
            llm = Gemini()
            embedding_model = FastEmbedEmbedding()
            Settings.llm = llm
            Settings.embed_model = embedding_model

            # Setup Qdrant
            client = qdrant_client.QdrantClient(location=":memory:")
            vector_store = QdrantVectorStore(
                collection_name="website",
                client=client,
                enable_hybrid=True,
                batch_size=20
            )

            # Load and index documents
            documents = SimpleWebPageReader(html_to_text=True).load_data(links)
            storage_context = StorageContext.from_defaults(vector_store=vector_store)
            index = VectorStoreIndex.from_documents(
                documents, storage_context=storage_context
            )
            query_engine = index.as_query_engine(vector_store_query_mode="hybrid")
            memory = ChatMemoryBuffer.from_defaults(token_limit=3000)
            chat_engine = index.as_chat_engine(
                chat_mode="context",
                memory=memory,
                system_prompt="You are an AI assistant who answers the user's questions."
            )
            chat_ready = True

# Chat Interface
if chat_ready:
    st.subheader("Ask Questions About the Website")
    user_query = st.text_input("Enter your question:")
    if st.button("Chat") and user_query:
        response = chat_engine.chat(user_query)
        st.write("**AI Assistant:**", response)
