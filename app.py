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
    # File IDs and their names
    file_ids = {
        "1vltJDB2dWuHz_oaNui-kqFzG14nRaNaL": "file1.ext",
        "131QwCzR1siIQSV2ugF1n2g737FVUCKAr": "file2.ext",
        "1VkfMF0XOEFNs9bmJaW7VT9pUlhwJjEUW": "file3.ext",
        "1iaJ7bkD1ln3AB4OzYSrP1hLl7wE2FwET": "file4.ext",
        "1wIltr1eGTQhmpNf9OU4lp-OCIqINbGFx": "file5.ext",
        "1g_6W4K_llIXmty0mhQcLc7PoINgGlhau": "file6.ext",
        "1qcqD5YRpTRt60iHaJCfSEqOR-xGdNEPu": "file7.ext",
        "1Wdv0J1zAx20e2uzs6E3Ph9g8nA_Jf8Wk": "file8.ext"
    }

    # Check if all files exist
    all_files_exist = all(os.path.exists(os.path.join(MODEL_PATH, fname)) for fname in file_ids.values())
    if not all_files_exist:
        # Download each file
        st.write("Downloading model files...")
        for file_id, fname in file_ids.items():
            download_file_from_google_drive(file_id, os.path.join(MODEL_PATH, fname))
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
