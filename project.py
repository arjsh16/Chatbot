
# My file that contains the api keys
import api_keys

# Importing gemini API dependencies
import google.generativeai as genai
import tensorflow_hub as hub
import numpy as np
# Importing Pinecone
from pinecone import Pinecone

# Library used for the webapp inerface
import streamlit as st

# initializing open AI and pinecone
genai.configure(api_key=api_keys.pwd.gemini())
model = genai.GenerativeModel("gemini-1.5-flash")
pc = Pinecone(api_key=api_keys.pwd.pinecone())
index = pc.Index(name="quickstart", host=api_keys.pwd.index())

# Adding title to the chatbox
st.title("Arjun's :orange[Chatbot]")

# Storing All messages as an array and displaying them
if 'messages' not in st.session_state:
  st.session_state.messages = []
for message in st.session_state.messages:
  st.chat_message(message['role']).markdown(message['content'])

# Taking Input
prompt=st.chat_input('Your Prompt goes here')

# Functions to embed and upload data to pinecone and retrieve it when needed
def get_embeddings(text):
    embeder = hub.load("https://tfhub.dev/google/universal-sentence-encoder/4")
    response = embeder([text])  # Generate embeddings
    return np.array(response)  # Convert to NumPy array and return the first embedding
def store_document(doc_id, text):
    index.upsert([(doc_id, get_embeddings(text), {"text": text})])  # Upload to Pinecone
def retrieve_context(query, top_k=5):
    query_embedding = get_embeddings(query)
    results = index.query(vector=query_embedding.tolist(), top_k=top_k, include_metadata=True)  # Use keyword arguments
    return [item['metadata']['text'] for item in results['matches']]

# Giving the output
if prompt:
  # Shows the message 
  st.chat_message('user').markdown(prompt)
  # Saves the previous messages 
  st.session_state.messages.append({'role':'user','content':prompt})

  # Retrieve context
  context = retrieve_context(prompt)  # Get relevant context
  context_str = "\n".join(context)  # Combine context into a single string
    
  # Generate response with context
  response = model.generate_content(context_str + "\n" + prompt)
  #if response.result and 'candidates' in response.result:
  if response._done and response.candidates:
    response_text =  response.candidates[0].content.parts[0].text
  else:
      response_text = "Sorry, we have run out of tokens."
  # show response generated
  st.chat_message('assistant').markdown(response_text)
  # save reponse so that the previous outputs of the chatbox can be seen
  st.session_state.messages.append({'role':'assistant','content':response_text})
