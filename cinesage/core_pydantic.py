import json

import streamlit as st
from dotenv import load_dotenv
from langchain_mistralai import ChatMistralAI
from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel
from typing import Optional, List
from langchain_core.output_parsers import PydanticOutputParser

# Load env
load_dotenv()

# Schema for structured output
class MovieInfo(BaseModel):
    title : str
    release_year : Optional[int]
    genre : List[str]
    director : Optional[str]
    cast : List[str]
    rating : Optional[float]
    plot_summary : str

# Prompt template
prompt = ChatPromptTemplate.from_messages([
    ("system",
        """
        Extract movie info from the paragraph {format_instructions}
        """,),
    ("human","{paragraph}"),
])

parser = PydanticOutputParser(pydantic_object=MovieInfo)

# Model
model = ChatMistralAI(model_name="mistral-small-2506", temperature=0.9)

# UI
st.set_page_config(page_title="CineSage", page_icon="🎬", layout="centered")
st.title("🎬 CineSage")
st.caption("Movie Information Extraction")

paragraph = st.text_area("Enter movie description", height=200)

if st.button("Extract"):
    if paragraph.strip() == "":
        st.warning("Please enter a movie description.")
    else:
        with st.spinner("Extracting..."):
            final_prompt = prompt.format_messages(
                paragraph=paragraph,
                format_instructions=parser.get_format_instructions(),
            )
            response = model.invoke(final_prompt)
            content = (
                response.content
                if isinstance(response.content, str)
                else json.dumps(response.content)
            )
            try:
                parsed = parser.parse(content)
                st.text_area(
                    "Extracted Information",
                    value=json.dumps(parsed.model_dump(), indent=2),
                    height=300,
                )
            except Exception:
                st.text_area("Extracted Information", value=content, height=300)