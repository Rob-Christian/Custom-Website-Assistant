# Website Chat Assistant

This application allows users to input a website URL. The app crawls all the links on the website that share the same root domain as the provided URL. It then uses OpenAI's GPT-powered Llama-Index to create a retrieval-based chain for answering questions about the website's content.

## Features
- **Website Crawling:** The app extracts and crawls all links that share the same root domain as the provided URL.
- **Content Processing:** After crawling, the app processes the content of the extracted links.
- **Question-Answering Interface:** Users can ask questions about the website's content, and the app will provide answers based on the indexed information.
- **Semantic Search:** Uses Llama-Index's embedding and retrieval capabilities for answering questions contextually.

## Requirements

Make sure you have the following Python packages installed:
- `streamlit`
- `requests`
- `beautifulsoup4`
- `llama_index`
- `openai`

You can install them using `pip`:

```bash
pip install streamlit requests beautifulsoup4 llama_index openai
```

## Setting up OpenAI API Key
Before running the app, you need to set your OpenAI API key.
1. Go to OpenAI API Keys.
2. Copy your API key.
3. In the streamlit_app.py file, set your API key as follows:
```bash
import os
os.environ["OPENAI_API_KEY"] = "your-api-key"
```
