import json

import streamlit as st
from dotenv import load_dotenv
from langchain_mistralai import ChatMistralAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import PydanticOutputParser
from pydantic import BaseModel
from typing import Optional, List

load_dotenv()


# pydantic schema — tells the LLM exactly what fields to return and their types
class MovieInfo(BaseModel):
    title: str
    release_year: Optional[int]
    genre: List[str]
    director: Optional[str]
    cast: List[str]
    rating: Optional[float]
    plot_summary: str


# parser reads the schema and generates instructions we inject into the prompt
parser = PydanticOutputParser(pydantic_object=MovieInfo)

# {format_instructions} gets replaced at runtime with the JSON schema the LLM must follow
prompt = ChatPromptTemplate.from_messages([
    ("system", "Extract movie info from the paragraph. {format_instructions}"),
    ("human", "{paragraph}"),
])

model = ChatMistralAI(name="mistral-small-2506", temperature=0.9)

st.set_page_config(page_title="CineSage", page_icon="🎬", layout="centered")
st.title("🎬 CineSage")
st.caption("Movie Information Extraction — Structured JSON Output")

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

            # response.content can sometimes be a list, make sure it's a string before parsing
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
                # if parsing fails just show the raw output
                st.text_area("Extracted Information", value=content, height=300)
