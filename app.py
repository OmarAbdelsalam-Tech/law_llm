import streamlit as st
import os
import requests
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig
from unittest.mock import patch

# Download the file from Google Drive
def download_file_from_google_drive(file_id, destination):
    base_url = "https://drive.google.com/uc?export=download"
    response = requests.get(f"{base_url}&id={file_id}", stream=True)
    
    with open(destination, "wb") as file:
        for chunk in response.iter_content(chunk_size=32768):
            file.write(chunk)

# Adjusted the model path to the current directory
MODEL_PATH = "/mount/src/law_llm"


@st.cache(allow_output_mutation=True, suppress_st_warning=True)
def load_model():
    # Check if model files exist
    if not os.path.exists(os.path.join(MODEL_PATH, "config.json")):
        # Create directory if it doesn't exist
        if not os.path.exists(MODEL_PATH):
            os.makedirs(MODEL_PATH)
        
        # Download each file
        st.write("Downloading model files...")
        download_file_from_google_drive("1qcqD5YRpTRt60iHaJCfSEqOR-xGdNEPu", os.path.join(MODEL_PATH, "config.json"))
        download_file_from_google_drive("1Wdv0J1zAx20e2uzs6E3Ph9g8nA_Jf8Wk", os.path.join(MODEL_PATH, "pytorch_model.bin"))
        download_file_from_google_drive("1vltJDB2dWuHz_oaNui-kqFzG14nRaNaL", os.path.join(MODEL_PATH, "vocab.json"))
        download_file_from_google_drive("1wIltr1eGTQhmpNf9OU4lp-OCIqINbGFx", os.path.join(MODEL_PATH, "merges.txt"))
        st.write("Model files downloaded!")

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
