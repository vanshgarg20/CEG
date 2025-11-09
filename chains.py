# chains.py
import os
from typing import Any, List

# Streamlit may not exist when running as pure API; so optional import
try:
    import streamlit as st
except Exception:
    st = None  # type: ignore

from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.exceptions import OutputParserException

load_dotenv()

def _get_secret(name: str) -> str | None:
    # Prefer Streamlit secrets, fallback to env
    val = None
    if st and hasattr(st, "secrets"):
        try:
            val = st.secrets.get(name)  # type: ignore[attr-defined]
        except Exception:
            pass
    return val or os.getenv(name)

class Chain:
    def __init__(self):
        api_key = _get_secret("GROQ_API_KEY")
        if not api_key:
            raise ValueError(
                "GROQ_API_KEY not set. Add it in Streamlit Secrets or as an environment variable."
            )

        # Support both param names across versions
        try:
            self.llm = ChatGroq(api_key=api_key, model_name="llama-3.1-8b-instant", temperature=0)
        except TypeError:
            self.llm = ChatGroq(groq_api_key=api_key, model_name="llama-3.1-8b-instant", temperature=0)

    def extract_jobs(self, cleaned_text: str) -> List[dict[str, Any]]:
        prompt_extract = PromptTemplate.from_template(
            """### SCRAPED TEXT FROM WEBSITE:
{page_data}
### INSTRUCTION:
The scraped text is from the careers page of a website.
Extract job postings and return JSON with keys: role, experience, skills, description.
Return only valid JSON (no preamble)."""
        )
        chain_extract = prompt_extract | self.llm
        res = chain_extract.invoke({"page_data": cleaned_text})
        try:
            res_json = JsonOutputParser().parse(getattr(res, "content", res))
        except OutputParserException:
            raise OutputParserException("Context too big or malformed model output.")
        return res_json if isinstance(res_json, list) else [res_json]

    def write_mail(self, job: dict, links: list[str]) -> str:
        prompt_email = PromptTemplate.from_template(
            """### JOB DESCRIPTION:
{job_description}

### INSTRUCTION:
You are Mohan, BDE at AtliQ (AI & Software Consulting). Write a concise, tailored cold email
showing how AtliQ can meet the needs above. Include the most relevant portfolio links: {link_list}.
No preamble. Output only the email in Markdown."""
        )
        chain_email = prompt_email | self.llm
        res = chain_email.invoke({"job_description": str(job), "link_list": links})
        return getattr(res, "content", str(res))
