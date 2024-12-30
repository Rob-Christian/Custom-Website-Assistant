import streamlit as st
from urllib.parse import urljoin, urlparse
from collections import deque
import requests
from bs4 import BeautifulSoup
from llama_index.readers.web import SimpleWebPageReader
from llama_index.llms.gemini import Gemini
from llama_index.embeddings.fastembed import FastEmbedEmbedding
import faiss
import numpy as np

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
            
            # Load documents
            documents = SimpleWebPageReader(html_to_text=True).load_data(links)

            # Generate embeddings
            texts = [doc.text for doc in documents]
            embeddings = np.array([embedding_model.embed(text) for text in texts])

            # Set up FAISS index
            dimension = embeddings.shape[1]
            index = faiss.IndexFlatL2(dimension)
            index.add(embeddings)

            # Store text for retrieval
            text_mapping = {i: text for i, text in enumerate(texts)}
            chat_ready = True

# Chat Interface
if chat_ready:
    st.subheader("Ask Questions About the Website")
    user_query = st.text_input("Enter your question:")
    if st.button("Chat") and user_query:
        # Query embedding
        query_embedding = np.array([embedding_model.embed(user_query)])
        _, indices = index.search(query_embedding, k=3)  # Top 3 results
        responses = [text_mapping[i] for i in indices[0]]
        
        # Display results
        for i, response in enumerate(responses, start=1):
            st.write(f"**Result {i}:**", response)
