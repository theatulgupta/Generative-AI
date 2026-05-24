import streamlit as st
from dotenv import load_dotenv
from langchain_mistralai import ChatMistralAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

load_dotenv()

prompt = ChatPromptTemplate.from_messages([
    (
        "system",
        """
You are a professional Movie Information Extraction Assistant.

Your task:
Extract structured information from a given movie paragraph and return it strictly in the specified format.

Rules:
- Do NOT add explanations
- Do NOT add extra commentary
- Follow the exact format
- If any information is missing, write NULL
- Keep answers concise and factual

Output Format:

Title:
Genre:
Director:
Main Cast:
Setting/Location:
Plot:
Themes:
Ratings:
Notable Features:

Short Summary:
"""
    ),
    (
        "human",
        "Extract information from the following paragraph:\n\n{paragraph}"
    ),
])

model = ChatMistralAI(name="mistral-small-2506", temperature=0.9)
parser = StrOutputParser()

# LCEL chain — paragraph goes in, formatted text comes out
chain = prompt | model | parser

st.set_page_config(page_title="CineSage", page_icon="🎬", layout="centered")
st.title("🎬 CineSage")
st.caption("Movie Information Extraction")

paragraph = st.text_area("Enter movie description", height=200)

if st.button("Extract"):
    if paragraph.strip() == "":
        st.warning("Please enter a movie description.")
    else:
        with st.spinner("Extracting..."):
            result = chain.invoke({"paragraph": paragraph})
            st.text_area("Extracted Information", value=result, height=300)
