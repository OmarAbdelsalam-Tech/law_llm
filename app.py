import streamlit as st
import os
import requests
from transformers import AutoModelForCausalLM, AutoTokenizer

# Download the model from Google Drive
def download_model(file_id, dest_folder):
    base_url = "https://drive.google.com/uc?export=download"
    response = requests.get(f"{base_url}&id={file_id}", stream=True)
    file_size = int(response.headers.get('content-length', 0))
    progress = st.progress(0)
    
    with open(os.path.join(dest_folder, "model.zip"), "wb") as file:
        for data in response.iter_content(chunk_size=1024):
            size = file.write(data)
            progress.progress(size/file_size)

    # Unzipping the model directory
    os.system(f"unzip {dest_folder}/model.zip -d {dest_folder}")
    st.write("Model downloaded and extracted!")

# Set up the model paths
MODEL_PATH = "./model"
DEST_FOLDER = "."

# Load the model and tokenizer
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
model = AutoModelForCausalLM.from_pretrained(MODEL_PATH)

st.title("Contract Law AI Assistant")
st.write("Ask any question about contract law:")

# Text input for the user's question
user_input = st.text_input("Your Question:")

if user_input:
    # Tokenize the user input and get model's answer
    input_ids = tokenizer.encode(user_input, return_tensors="pt")
    generated_ids = model.generate(input_ids, max_length=100, num_return_sequences=1)
    answer = tokenizer.decode(generated_ids[0], skip_special_tokens=True)
    
    # Display the model's answer
    st.write("Answer:", answer)

# Provide a button to download the model (you can place this anywhere in your app)
if st.button("Download Model"):
    FILE_ID = "1KBas6Rux5tTjxFI2ZuweXsVYphSneSdd"
    download_model(FILE_ID, DEST_FOLDER)
