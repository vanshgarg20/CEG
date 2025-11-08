# app.py  — Flask API for your Cold Email Generator
import os
import re
from datetime import datetime
from html import escape
from typing import Any, List

from flask import Flask, request, jsonify
from werkzeug.middleware.proxy_fix import ProxyFix

from langchain_community.document_loaders import WebBaseLoader

from chains import Chain
from portfolio import Portfolio
from utils import clean_text

# ---------- helpers ----------
URL_RE = re.compile(r"^https?://", re.IGNORECASE)

def fetch_and_clean(url: str) -> str:
    doc = WebBaseLoader(url).load()[0]
    return clean_text(doc.page_content)

def normalize_skills(raw: Any) -> List[str]:
    if isinstance(raw, str):
        return [s.strip() for s in raw.split(",") if s.strip()]
    if isinstance(raw, list):
        return [str(s).strip() for s in raw if str(s).strip()]
    return [str(raw)]

def download_name(prefix="email"):
    return f"{prefix}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.md"

# ---------- app + singletons ----------
app = Flask(__name__)
app.wsgi_app = ProxyFix(app.wsgi_app)

# Lazy-init so cold starts are fast
_chain: Chain | None = None
_portfolio: Portfolio | None = None

def get_chain() -> Chain:
    global _chain
    if _chain is None:
        _chain = Chain()
    return _chain

def get_portfolio() -> Portfolio:
    global _portfolio
    if _portfolio is None:
        _portfolio = Portfolio()
        _portfolio.load_portfolio()
    return _portfolio

# ---------- routes ----------
@app.get("/health")
def health():
    return jsonify(ok=True)

@app.post("/generate")
def generate():
    """
    JSON body:
    {
      "url": "https://company.com/careers/job",
      "tone": "Confident",            # optional
      "cta": "Request an Interview"   # optional
    }
    """
    data = request.get_json(silent=True) or {}
    url = (data.get("url") or "").strip()
    tone = data.get("tone") or "Confident"
    cta = data.get("cta") or "Request an Interview"

    if not url or not URL_RE.match(url):
        return jsonify(error="Provide a valid http(s) URL in 'url'."), 400

    try:
        text = fetch_and_clean(url)
        chain = get_chain()
        portfolio = get_portfolio()   # currently unused in email, but kept for parity

        jobs = chain.extract_jobs(text) or []
        results = []

        for job in jobs:
            role  = job.get("role", "Software Engineer")
            desc  = job.get("description", "")
            exp   = job.get("experience", "N/A")
            skills = normalize_skills(job.get("skills", []))

            job_with_prefs = {**job, "tone": tone, "cta": cta}
            email_md = chain.write_mail(job_with_prefs, [])  # portfolio links list is [] per your UI

            results.append({
                "role": role,
                "description": desc,
                "experience": exp,
                "skills": skills,
                "email_markdown": email_md,
                "download_name": download_name()
            })

        return jsonify(
            url=url,
            count=len(results),
            results=results
        )
    except Exception as e:
        return jsonify(error=str(e)), 500

# Optional: simple landing
@app.get("/")
def index():
    return jsonify(
        message="Cold Email Generator API",
        endpoints={
            "POST /generate": {
                "body": {"url": "https://...", "tone": "Confident", "cta": "Request an Interview"}
            },
            "GET /health": {}
        }
    )

if __name__ == "__main__":
    # Local dev: python app.py
    port = int(os.getenv("PORT", "8080"))
    app.run(host="0.0.0.0", port=port, debug=True)
