from dotenv import load_dotenv
import os
import requests
from io import BytesIO
from typing import Set
from PIL import Image
import streamlit as st
from streamlit_chat import message
from core import query_llm

load_dotenv()

st.set_page_config(
    page_title="AsyncAPI Chat Assistant",
    page_icon="🦜",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Function to format sources
def create_sources_string(source_urls: Set[str]) -> str:
    if not source_urls:
        return ""
    sources_list = sorted(list(source_urls))
    return "Sources:\n" + "\n".join(f"{i+1}. {src}" for i, src in enumerate(sources_list))

# Function to get a profile picture from Gravatar
def get_profile_picture(email: str):
    gravatar_url = f"https://www.gravatar.com/avatar/{hash(email)}?d=identicon&s=200"
    response = requests.get(gravatar_url)
    return Image.open(BytesIO(response.content))

# Custom CSS for dark mode styling
st.markdown(
    """
    <style>
        .stApp { background-color: #1E1E1E; color: #FFFFFF; }
        .stTextInput > div > div > input { background-color: #2D2D2D; color: #FFFFFF; }
        .stButton > button { background-color: #4CAF50; color: #FFFFFF; }
        .stSidebar { background-color: #252526; }
        .stMessage { background-color: #2D2D2D; }
    </style>
    """,
    unsafe_allow_html=True,
)

# Sidebar with user info
with st.sidebar:
    st.title("User Profile")
    user_name = "Manish Sharma"
    user_email = "manishsharma@gmail.com"
    st.image(get_profile_picture(user_email), width=150)
    st.write(f"**Name:** {user_name}")
    st.write(f"**Email:** {user_email}")

st.header("AsyncAPI 🦜🔗 Assistant")

# Initialize session state
if "chat_answers_history" not in st.session_state:
    st.session_state["chat_answers_history"] = []
    st.session_state["user_prompt_history"] = []
    st.session_state["chat_history"] = []

# Layout with two columns
col1, col2 = st.columns([2, 1])

with col1:
    prompt = st.text_input("Prompt", placeholder="Enter your AsyncAPI-related question...")

with col2:
    if st.button("Submit", key="submit"):
        prompt = prompt or "Hello"  # Default message if input is empty

if prompt:
    with st.spinner("Generating response..."):
        generated_response = query_llm(
            query=prompt,
            chat_history=st.session_state["chat_history"]
        )

        # Extract sources from the response
        sources = set()
        if "sources" in generated_response:
            for source in generated_response["sources"]:
                if "source" in source:
                    sources.add(source["source"])
        
        formatted_response = (
            f"{generated_response['answer']} \n\n {create_sources_string(sources)}"
        )

        st.session_state["user_prompt_history"].append(prompt)
        st.session_state["chat_answers_history"].append(formatted_response)
        st.session_state["chat_history"].append(("human", prompt))
        st.session_state["chat_history"].append(("ai", generated_response["answer"]))

# Display chat history
if st.session_state["chat_answers_history"]:
    for user_query, generated_response in zip(
        st.session_state["user_prompt_history"],
        st.session_state["chat_answers_history"],
    ):
        message(user_query, is_user=True, key=f"user_{hash(user_query)}")
        message(generated_response, key=f"bot_{hash(generated_response)}")

# Footer
st.markdown("---")
st.markdown("Powered by LangChain and Streamlit")