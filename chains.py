# chains.py
import os
import time
from typing import Any, List

from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.exceptions import OutputParserException

# Optional import (only if running under Streamlit)
try:
    import streamlit as st
except Exception:
    st = None

load_dotenv()


# --------------------- Helper: Secrets ---------------------
def _get_secret(name: str) -> str | None:
    """Read from Streamlit secrets (if present) or OS env."""
    val = None
    if st and hasattr(st, "secrets"):
        try:
            val = st.secrets.get(name)
        except Exception:
            pass
    return val or os.getenv(name)


# --------------------- Helper: Create Groq LLM ---------------------
def _make_groq_llm(api_key: str, model_name: str, temperature: float = 0.0):
    """Create a Groq model (with backward compatibility)."""
    kwargs = dict(model_name=model_name, temperature=temperature)
    try:
        return ChatGroq(api_key=api_key, **kwargs)
    except TypeError:
        return ChatGroq(groq_api_key=api_key, **kwargs)


# --------------------- Helper: Create Gemini LLM ---------------------
def _make_gemini_llm(api_key: str, model_name: str = "gemini-1.5-flash", temperature: float = 0.0):
    """Create a Gemini model."""
    return ChatGoogleGenerativeAI(model=model_name, google_api_key=api_key, temperature=temperature)


# --------------------- Retry Wrapper ---------------------
def _invoke_with_retry(chain, payload, retries: int = 1, backoff_sec: float = 3.0):
    """Retry for transient rate-limit (429) or throttling messages."""
    last = None
    for _ in range(retries + 1):
        try:
            return chain.invoke(payload)
        except Exception as e:
            msg = str(e).lower()
            last = e
            if any(k in msg for k in ["rate limit", "429", "quota", "exceeded"]):
                time.sleep(backoff_sec)
                continue
            raise
    raise last


# --------------------- Main Chain Class ---------------------
class Chain:
    """
    - extract_jobs → prefers Groq 8B → fallback Groq 70B → Gemini
    - write_mail  → prefers Groq 70B → fallback Groq 8B → Gemini
    Supports:
        GROQ_API_KEY
        GROQ_API_KEY_FAST
        GROQ_API_KEY_HEAVY
        GEMINI_API_KEY
    """

    def __init__(self):
        # Load keys
        base_key = _get_secret("GROQ_API_KEY")
        gemini_key = _get_secret("GEMINI_API_KEY")

        if not base_key and not gemini_key:
            raise ValueError("❌ No API key found. Add GROQ_API_KEY or GEMINI_API_KEY in secrets or .env")

        self.fast_key = _get_secret("GROQ_API_KEY_FAST") or base_key
        self.heavy_key = _get_secret("GROQ_API_KEY_HEAVY") or base_key
        self.gemini_key = gemini_key

        # Models
        self.fast_model = "llama-3.1-8b-instant"
        self.heavy_model = "llama-3.3-70b-versatile"
        self.gemini_model = "gemini-1.5-flash"

        # Parser
        self._json_parser = JsonOutputParser()

    # -------------------- EXTRACT --------------------
    def extract_jobs(self, cleaned_text: str) -> List[dict[str, Any]]:
        """Extract job details using Groq first, fallback to Gemini."""
        prompt = PromptTemplate.from_template(
            """### SCRAPED TEXT FROM WEBSITE:
{page_data}

### INSTRUCTION:
The scraped text is from a careers page.
Extract job postings and return valid JSON with keys: role, experience, skills, description.
Return ONLY JSON (no extra text)."""
        )

        attempts = []

        # Groq models
        if self.fast_key:
            attempts.append(("groq", self.fast_key, self.fast_model))
        if self.heavy_key:
            attempts.append(("groq", self.heavy_key, self.heavy_model))

        # Gemini fallback
        if self.gemini_key:
            attempts.append(("gemini", self.gemini_key, self.gemini_model))

        last_error = None
        for provider, api_key, model in attempts:
            try:
                if provider == "groq":
                    llm = _make_groq_llm(api_key, model)
                else:
                    llm = _make_gemini_llm(api_key, model)
                chain = prompt | llm
                res = _invoke_with_retry(chain, {"page_data": cleaned_text}, retries=1)
                parsed = self._json_parser.parse(getattr(res, "content", res))
                return parsed if isinstance(parsed, list) else [parsed]
            except Exception as e:
                last_error = e
                continue

        raise last_error or RuntimeError("❌ Extraction failed on all providers.")

    # -------------------- WRITE EMAIL --------------------
    def write_mail(self, job: dict, links: List[str]) -> str:
        """Generate cold email using Groq first, fallback to Gemini."""
        prompt = PromptTemplate.from_template(
            """### JOB DESCRIPTION:
{job_description}

### INSTRUCTION:
You are Mohan, BDE at AtliQ (AI & Software Consulting).
Write a short, persuasive cold email to the company.
Include relevant portfolio links: {link_list}.
Return ONLY the email in Markdown, no preamble."""
        )

        attempts = []

        # Groq heavy first, then fast
        if self.heavy_key:
            attempts.append(("groq", self.heavy_key, self.heavy_model))
        if self.fast_key:
            attempts.append(("groq", self.fast_key, self.fast_model))

        # Gemini fallback
        if self.gemini_key:
            attempts.append(("gemini", self.gemini_key, self.gemini_model))

        last_error = None
        for provider, api_key, model in attempts:
            try:
                if provider == "groq":
                    llm = _make_groq_llm(api_key, model)
                else:
                    llm = _make_gemini_llm(api_key, model)
                chain = prompt | llm
                res = _invoke_with_retry(
                    chain,
                    {"job_description": str(job), "link_list": links},
                    retries=1,
                )
                return getattr(res, "content", str(res))
            except Exception as e:
                last_error = e
                continue

        raise last_error or RuntimeError("❌ Email writing failed on all providers.")
