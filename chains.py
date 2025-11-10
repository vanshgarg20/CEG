# chains.py
import os
import time
from typing import Any, List

from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.exceptions import OutputParserException
from langchain_groq import ChatGroq

# Gemini is optional; handle if the package isn't installed
try:
    from langchain_google_genai import ChatGoogleGenerativeAI
    _HAS_GEMINI = True
except Exception:
    ChatGoogleGenerativeAI = None  # type: ignore
    _HAS_GEMINI = False

# Optional Streamlit (for secrets in cloud)
try:
    import streamlit as st  # type: ignore
except Exception:
    st = None  # type: ignore

load_dotenv()


# --------------------- Secrets helpers ---------------------
def _get_secret(name: str) -> str | None:
    """Read from Streamlit secrets if present, else OS env."""
    val = None
    if st and hasattr(st, "secrets"):
        try:
            val = st.secrets.get(name)  # type: ignore[attr-defined]
        except Exception:
            pass
    return val or os.getenv(name)


# --------------------- LLM builders ---------------------
def _make_groq_llm(api_key: str, model_name: str, temperature: float = 0.0):
    """Create Groq LLM (supports old/new arg names)."""
    kwargs = dict(model_name=model_name, temperature=temperature)
    try:
        return ChatGroq(api_key=api_key, **kwargs)
    except TypeError:
        return ChatGroq(groq_api_key=api_key, **kwargs)


def _make_gemini_llm(api_key: str, model_name: str = "gemini-1.5-flash", temperature: float = 0.0):
    """Create Gemini LLM if available."""
    if not _HAS_GEMINI:
        raise RuntimeError("langchain-google-genai is not installed.")
    return ChatGoogleGenerativeAI(model=model_name, google_api_key=api_key, temperature=temperature)


# --------------------- Retry wrapper ---------------------
def _invoke_with_retry(chain, payload, retries: int = 1, backoff_sec: float = 3.0):
    last_err = None
    for _ in range(retries + 1):
        try:
            return chain.invoke(payload)
        except Exception as e:
            msg = str(e).lower()
            last_err = e
            if any(k in msg for k in ("429", "rate limit", "quota", "exceeded", "temporarily")):
                time.sleep(backoff_sec)
                continue
            raise
    raise last_err


# --------------------- Main chain ---------------------
class Chain:
    """
    - extract_jobs → Groq 8B → Groq 70B → Gemini (if key + package are available)
    - write_mail  → Groq 70B → Groq 8B → Gemini
    Secrets supported:
      GROQ_API_KEY (base), GROQ_API_KEY_FAST, GROQ_API_KEY_HEAVY, GEMINI_API_KEY
    """

    def __init__(self):
        base_key = _get_secret("GROQ_API_KEY")
        self.fast_key = _get_secret("GROQ_API_KEY_FAST") or base_key
        self.heavy_key = _get_secret("GROQ_API_KEY_HEAVY") or base_key
        self.gemini_key = _get_secret("GEMINI_API_KEY")

        if not (self.fast_key or self.heavy_key or self.gemini_key):
            raise ValueError("❌ No API key found. Add GROQ_API_KEY or GEMINI_API_KEY in secrets or .env")

        self.fast_model = "llama-3.1-8b-instant"
        self.heavy_model = "llama-3.3-70b-versatile"
        self.gemini_model = "gemini-1.5-flash"
        self._json_parser = JsonOutputParser()

    # -------- extract --------
    def extract_jobs(self, cleaned_text: str) -> List[dict[str, Any]]:
        prompt_extract = PromptTemplate.from_template(
            """### SCRAPED TEXT FROM WEBSITE:
{page_data}

### INSTRUCTION:
The scraped text is from a careers page.
Extract job postings and return valid JSON with keys: role, experience, skills, description.
Return ONLY JSON (no extra text)."""
        )

        attempts: list[tuple[str, str, str]] = []
        if self.fast_key:
            attempts.append(("groq", self.fast_key, self.fast_model))
        if self.heavy_key:
            attempts.append(("groq", self.heavy_key, self.heavy_model))
        if self.gemini_key and _HAS_GEMINI:
            attempts.append(("gemini", self.gemini_key, self.gemini_model))

        last_error = None
        for provider, api_key, model in attempts:
            try:
                llm = _make_groq_llm(api_key, model) if provider == "groq" else _make_gemini_llm(api_key, model)
                chain_extract = prompt_extract | llm
                res = _invoke_with_retry(chain_extract, {"page_data": cleaned_text}, retries=1)
                content = getattr(res, "content", res)
                parsed = self._json_parser.parse(content)
                return parsed if isinstance(parsed, list) else [parsed]
            except Exception as e:
                last_error = e
                continue

        raise last_error or RuntimeError("Extraction failed on all providers.")

    # -------- write email (plain text only) --------
    def write_mail(self, job: dict, links: List[str]) -> str:
        # tone/cta may be passed inside job, but we generate plain text regardless.
        prompt_email = PromptTemplate.from_template(
            """### JOB DESCRIPTION:
{job_description}

### INSTRUCTION:
You are Mohan, BDE at AtliQ (AI & Software Consulting).
Write a SHORT, well-structured PLAIN TEXT cold email (no markdown, no hashtags, no asterisks).
Include:
- Subject line (single concise line starting with "Subject:")
- Greeting
- One-paragraph intro about AtliQ and why we’re relevant to this job
- A specific value proposition referencing the role/requirements
- A clear call to action to schedule a quick call
- Polite sign-off with name and title

Return ONLY clean plain text."""
        )

        attempts: list[tuple[str, str, str]] = []
        if self.heavy_key:
            attempts.append(("groq", self.heavy_key, self.heavy_model))
        if self.fast_key:
            attempts.append(("groq", self.fast_key, self.fast_model))
        if self.gemini_key and _HAS_GEMINI:
            attempts.append(("gemini", self.gemini_key, self.gemini_model))

        last_error = None
        for provider, api_key, model in attempts:
            try:
                llm = _make_groq_llm(api_key, model) if provider == "groq" else _make_gemini_llm(api_key, model)
                chain_email = prompt_email | llm
                res = _invoke_with_retry(chain_email, {"job_description": str(job)}, retries=1)
                return getattr(res, "content", str(res))
            except Exception as e:
                last_error = e
                continue

        raise last_error or RuntimeError("Email writing failed on all providers.")
