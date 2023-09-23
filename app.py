import streamlit as st
import os
import requests
import zipfile
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig
from unittest.mock import patch

# Download the file from Google Drive
def download_file_from_google_drive(file_id, destination):
    base_url = "https://drive.google.com/uc?export=download"
    response = requests.get(f"{base_url}&id={file_id}", stream=True)
    
    with open(destination, "wb") as file:
        for chunk in response.iter_content(chunk_size=32768):
            file.write(chunk)

# Extract a ZIP file
def extract_zip(zip_path, extract_to="."):
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(extract_to)

# Set up the model paths
MODEL_PATH = "./"

@st.cache(allow_output_mutation=True, suppress_st_warning=True)
def load_model():
    # Check if model files exist
    if not os.path.exists(os.path.join(MODEL_PATH, "config.json")):
        # Create directory if it doesn't exist
        if not os.path.exists(MODEL_PATH):
            os.makedirs(MODEL_PATH)
        
        # Download ZIP file
        st.write("Downloading model files...")
        zip_path = os.path.join(MODEL_PATH, "model_files.zip")
        download_file_from_google_drive("19INw23gJi5kTb9T5yOczIlzGID2Elrf9", zip_path)
        
        # Extract the ZIP file
        extract_zip(zip_path, MODEL_PATH)
        
        st.write("Model files extracted!")

    # Load the model and tokenizer using AutoConfig
    config = AutoConfig.from_pretrained(MODEL_PATH)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, config=config)
    model = AutoModelForCausalLM.from_pretrained(MODEL_PATH, config=config)
    
    return model, tokenizer

def main():
    st.title("Contract Law AI Assistant")
    st.write("Ask any question about contract law:")
    
    # Load the model and tokenizer
    model, tokenizer = load_model()

    # Text input for the user's question
    user_input = st.text_input("Your Question:")

    if user_input:
        with st.spinner('Generating response...'):
            # Tokenize the user input and get model's answer
            input_ids = tokenizer.encode(user_input, return_tensors="pt")
            generated_ids = model.generate(input_ids, max_length=100, num_return_sequences=1)
            answer = tokenizer.decode(generated_ids[0], skip_special_tokens=True)
    
        # Display the model's answer
        st.write("Answer:", answer)

with patch("transformers.dynamic_module_utils.resolve_trust_remote_code", lambda *args, **kwargs: False):
    if __name__ == "__main__":
        main()
