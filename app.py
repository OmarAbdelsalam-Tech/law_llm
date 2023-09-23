import streamlit as st
import openai

openai.api_key = st.secrets["openai"]["api_key"]



def main():
    st.set_page_config(page_title="Law AI Assistant", page_icon=":guardsman:", layout="wide")
    st.title("Sheikh-GPT")
    st.subheader(" By: Omar Abdelsalam")

    # Set up the layout
    col1, col2 = st.columns([2, 1])
    with col1:
        st.image(background_image_path, width=500, use_column_width=False)

        # Wrap the chat history label in a ScrollView widget
        scroll_view = st.empty()
        chat_history = scroll_view.markdown("", unsafe_allow_html=True)

        user_input = st.text_input("Type your message", key="user_input")

        if st.button("Send"):
            send_message(user_input, chat_history)



def send_message(user_input, chat_history):
    message = user_input.strip()
    if message:
        chat_history.write(f"<p style='font-weight:bold'>User: {message}</p>", unsafe_allow_html=True)

        response_text = get_chatbot_response(message)
        response = f"<p style='color:blue;font-weight:bold'>Sheikh-GPT: {response_text}</p>"
        chat_history.write(response, unsafe_allow_html=True)

        # Clear the user input after sending the message
        user_input = ""




def get_chatbot_response(message):
    openai_response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "You are a knowledgeable Law Teacher."},
                {"role": "user", "content": message},
            ],
        )
    
    
    
    

    assistant_message = openai_response.choices[0]["message"]["content"]
    return assistant_message.strip()


if __name__ == "__main__":
    main()
