# Import necessary libraries
import google.generativeai as genai
import tensorflow_hub as hub
import numpy as np
from pinecone import Pinecone
import streamlit as st

# Initialize APIs
genai.configure('Your gemini api key goes here')
model = genai.GenerativeModel("gemini-1.5-flash")
pc = Pinecone('Your Pinecone API key goes here')
index = pc.Index('your vector database details go here')

# Streamlit interface setup
st.title("Chat :orange[Bot]")
if 'messages' not in st.session_state:
    st.session_state.messages = []

# Display chat history
for message in st.session_state.messages:
    st.chat_message(message['role']).markdown(message['content'])

# Take user input
prompt = st.chat_input("Your Prompt goes here")

# Embedding and retrieval functions
def get_embeddings(text):
    embeder = hub.load("https://tfhub.dev/google/universal-sentence-encoder/4")
    return np.array(embeder([text]))

def store_document(doc_id, text):
    index.upsert([(doc_id, get_embeddings(text), {"text": text})])

def retrieve_context(query, top_k=5):
    query_embedding = get_embeddings(query)
    results = index.query(vector=query_embedding.tolist(), top_k=top_k, include_metadata=True)
    return [item['metadata']['text'] for item in results['matches']]

# Generate and display response
if prompt:
    st.chat_message("user").markdown(prompt)
    st.session_state.messages.append({"role": "user", "content": prompt})
    context = retrieve_context(prompt)
    context_str = "\n".join(context)
    response = model.generate_content(context_str + "\n" + prompt)
    if response._done and response.candidates:
        response_text = response.candidates[0].content.parts[0].text
    else:
        response_text = "Sorry, There was some unexpected error."
    st.chat_message("assistant").markdown(response_text)
    st.session_state.messages.append({"role": "assistant", "content": response_text})
