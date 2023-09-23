import streamlit as st
import os
import requests
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig

# Check if 'model_loaded' is in the session state
if 'model_loaded' not in st.session_state:
    st.session_state.model_loaded = False

# Download the file from Google Drive
def download_file_from_google_drive(file_id, destination):
    base_url = "https://drive.google.com/uc?export=download"
    response = requests.get(f"{base_url}&id={file_id}", stream=True)
    
    with open(destination, "wb") as file:
        for chunk in response.iter_content(chunk_size=32768):
            file.write(chunk)

# Set up the model paths
MODEL_PATH = "./lawllm"
DEST_FOLDER = "."

if not os.path.exists(MODEL_PATH):
    os.makedirs(MODEL_PATH)

st.title("Contract Law AI Assistant")
st.write("Ask any question about contract law:")

if not st.session_state.model_loaded:
    # Download each file
    st.write("Downloading model files...")
    download_file_from_google_drive("1qcqD5YRpTRt60iHaJCfSEqOR-xGdNEPu", os.path.join(MODEL_PATH, "config.json"))
    download_file_from_google_drive("1Wdv0J1zAx20e2uzs6E3Ph9g8nA_Jf8Wk", os.path.join(MODEL_PATH, "pytorch_model.bin"))
    download_file_from_google_drive("1vltJDB2dWuHz_oaNui-kqFzG14nRaNaL", os.path.join(MODEL_PATH, "vocab.json"))
    download_file_from_google_drive("1wIltr1eGTQhmpNf9OU4lp-OCIqINbGFx", os.path.join(MODEL_PATH, "merges.txt"))
    st.write("Model files downloaded!")

    # Load the model and tokenizer
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
    model = AutoModelForCausalLM.from_pretrained(MODEL_PATH)
    st.session_state.model_loaded = True
else:
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
    model = AutoModelForCausalLM.from_pretrained(MODEL_PATH)

# Text input for the user's question
user_input = st.text_input("Your Question:")

if user_input and st.session_state.model_loaded:
    with st.spinner('Generating response...'):
        # Tokenize the user input and get model's answer
        input_ids = tokenizer.encode(user_input, return_tensors="pt")
        generated_ids = model.generate(input_ids, max_length=100, num_return_sequences=1)
        answer = tokenizer.decode(generated_ids[0], skip_special_tokens=True)
    
    # Display the model's answer
    st.write("Answer:", answer)
