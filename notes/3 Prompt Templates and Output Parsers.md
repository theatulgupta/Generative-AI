# 3. Prompt Templates & Output Parsers

---

## Prompt Templates

Instead of hardcoding prompts:

```text
"Explain Runnables in simple words"
```

use templates with variables:

```text
"Explain {topic} in simple words"
```

This makes prompts:
- reusable
- dynamic
- clean

---

## ChatPromptTemplate

Most common template for chat models.

### From Template (Single User Message)

```python
from langchain_core.prompts import ChatPromptTemplate

prompt = ChatPromptTemplate.from_template(
    "Explain {topic} in simple words"
)
```

### From Messages (System + User)

```python
prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a code generator"),
    ("user", "{topic}")
])
```

Roles:
- `"system"` → sets AI behavior
- `"user"` → human input
- `"assistant"` → AI response (for few-shot)

---

## Formatting a Prompt

```python
formatted = prompt.format_messages(topic="Runnables in LangChain")
```

Returns a list of message objects ready to pass to the model.

---

## Output Parsers

LLM returns an `AIMessage` object.

Output parsers extract/transform the content.

```text
AIMessage → Parser → Usable Output
```

---

## StrOutputParser

Extracts just the string content from `AIMessage`.

```python
from langchain_core.output_parsers import StrOutputParser

parser = StrOutputParser()
result = parser.parse(response)  # returns plain string
```

Used in almost every chain for clean text output.

---

## PydanticOutputParser

Parses LLM output into a structured Pydantic model.

```text
LLM JSON string → Validated Python Object
```

### Step 1: Define Schema

```python
from pydantic import BaseModel
from typing import Optional, List

class MovieInfo(BaseModel):
    title: str
    release_year: Optional[int]
    genre: List[str]
    director: Optional[str]
    cast: List[str]
    rating: Optional[float]
    plot_summary: str
```

### Step 2: Create Parser

```python
from langchain_core.output_parsers import PydanticOutputParser

parser = PydanticOutputParser(pydantic_object=MovieInfo)
```

### Step 3: Inject Format Instructions into Prompt

```python
prompt = ChatPromptTemplate.from_messages([
    ("system", "Extract movie info from the paragraph {format_instructions}"),
    ("human", "{paragraph}"),
])

final_prompt = prompt.format_messages(
    paragraph=paragraph,
    format_instructions=parser.get_format_instructions(),
)
```

`get_format_instructions()` tells the LLM exactly what JSON format to return.

### Step 4: Parse Response

```python
parsed = parser.parse(response.content)
print(parsed.title)
print(parsed.genre)
print(parsed.model_dump())  # dict
```

---

## Pydantic Basics

Pydantic is a Python library for:

```text
Data validation using type hints
```

### BaseModel

```python
from pydantic import BaseModel
from typing import Optional, List

class MovieInfo(BaseModel):
    title: str
    release_year: Optional[int]
    genre: List[str]
```

- `str` → required string
- `Optional[int]` → nullable integer
- `List[str]` → list of strings

### Why Pydantic?

```text
LLM output is just text
```

Pydantic ensures:
- correct types
- required fields present
- easy to use as Python object

### `.model_dump()`

Converts Pydantic object to Python dict.

```python
parsed.model_dump()
# {"title": "Inception", "genre": ["Sci-Fi"], ...}
```

---

## CineSage Project

A Streamlit app that extracts structured movie info from a paragraph.

### core.py (Plain Text)

```text
Paragraph → Prompt → Mistral → StrOutputParser → Formatted Text
```

Uses a detailed system prompt with output format instructions written manually.

### core_pydantic.py (Structured JSON)

```text
Paragraph → Prompt + format_instructions → Mistral → PydanticOutputParser → MovieInfo object
```

Uses `PydanticOutputParser` to get validated structured data.

---

## Important Interview Points

- `ChatPromptTemplate` makes prompts reusable and dynamic
- `format_messages()` returns list of message objects
- `StrOutputParser` extracts plain string from AIMessage
- `PydanticOutputParser` validates and structures LLM output
- `get_format_instructions()` injects JSON schema into prompt
- Pydantic `Optional` fields won't fail if LLM returns null
- `.model_dump()` converts Pydantic object to dict
