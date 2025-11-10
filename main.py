# main.py
import re
from html import escape
from datetime import datetime
from typing import Any, List

import streamlit as st
from langchain_community.document_loaders import WebBaseLoader

from chains import Chain
from portfolio import Portfolio
from utils import clean_text

# --------------------- PAGE CONFIG ---------------------
st.set_page_config(page_title="Cold Email Generator", page_icon="📧", layout="wide")

# --------------------- ONE-TIME STATE ---------------------
if "chain" not in st.session_state:
    st.session_state["chain"] = Chain()

if "portfolio" not in st.session_state:
    csv_path = "my_portfolio.csv"  # repo root
    st.session_state["portfolio"] = Portfolio(csv_path=csv_path)
    with st.spinner("📚 Loading your portfolio…"):
        st.session_state["portfolio"].load_portfolio()
    st.toast("Portfolio loaded", icon="📁")

chain: Chain = st.session_state["chain"]
portfolio: Portfolio = st.session_state["portfolio"]

# --------------------- STYLES (base + responsive + copy button) ---------------------
st.markdown(
    """
<style>
/* base layout */
.block-container{max-width:1200px;padding-top:3.75rem;padding-bottom:3rem;overflow:visible}
@keyframes pulseGradient{0%{background-position:0% 50%}50%{background-position:100% 50%}100%{background-position:0% 50%}}
.hero-wrap{display:inline-flex;align-items:center;gap:.9rem;margin:.25rem auto .4rem;position:relative;left:50%;transform:translateX(-50%);overflow:visible}
.hero-logo{width:56px;height:56px;padding:6px;border-radius:14px;display:inline-flex;align-items:center;justify-content:center;background:linear-gradient(135deg,#00c6ff,#7b61ff,#ff6ec7);background-size:200% 200%;animation:pulseGradient 6s ease infinite;box-shadow:0 10px 24px rgba(0,0,0,.25)}
.hero-logo svg{width:32px;height:32px;fill:#fff;filter:drop-shadow(0 0 4px rgba(255,255,255,.35))}
.hero-title{font-size:3rem;font-weight:800;line-height:1.12;background:linear-gradient(90deg,#00c6ff,#7b61ff,#ff6ec7);-webkit-background-clip:text;-webkit-text-fill-color:transparent;background-size:220% 220%;animation:pulseGradient 8s ease infinite;white-space:nowrap}
.hero-sub{text-align:center;font-size:1.12rem;color:#cfcfcf;margin:0 0 1.6rem}
.card{background:rgba(255,255,255,.05);border:1px solid rgba(255,255,255,.08);border-radius:18px;padding:1.2rem 1.4rem;box-shadow:0 12px 30px rgba(0,0,0,.25);backdrop-filter:blur(8px)}
.chip{display:inline-block;padding:.3rem .65rem;margin:.18rem .3rem .18rem 0;border-radius:999px;border:1px solid rgba(255,255,255,.15);font-size:.85rem}
.badge{display:inline-block;padding:.25rem .55rem;border-radius:8px;background:rgba(255,255,255,.1);border:1px solid rgba(255,255,255,.15);font-size:.8rem;margin-left:.35rem;vertical-align:middle}
hr{border:none;height:1px;background:linear-gradient(90deg,transparent,rgba(255,255,255,.2),transparent)}
.stTextInput > div > div > input{height:3rem;font-size:1rem}
pre, pre code { white-space: pre-wrap !important; word-break: break-word !important }

/* copyable plain email block */
.plain-email{margin-top:.5rem}
.plain-email .email-toolbar{display:flex;justify-content:flex-end;gap:.5rem;margin-bottom:.4rem}
.copy-btn{
  border:1px solid rgba(255,255,255,.15);
  background:rgba(255,255,255,.08);
  padding:.35rem .7rem;border-radius:8px;cursor:pointer;
}
.copy-btn:active{transform:translateY(1px)}
.plain-email textarea{
  width:100%;height:300px; /* keep scrollbar */
  border:1px solid rgba(255,255,255,.15);
  background:rgba(255,255,255,.03);
  color:inherit;border-radius:12px;padding:.75rem;
  font: 0.92rem/1.4 ui-monospace, SFMono-Regular, Menlo, Consolas, "Liberation Mono", monospace;
  resize:vertical; /* user can resize; scrollbar stays */
}

/* tablets */
@media (max-width: 900px){
  .block-container{max-width:100%;padding-top:2.9rem;padding-left:1rem;padding-right:1rem}
  .hero-logo{width:48px;height:48px;padding:5px}
  .hero-logo svg{width:28px;height:28px}
  .hero-title{font-size:2.2rem;white-space:normal}
  .hero-sub{font-size:1rem}
  .card{padding:1rem}
  .chip{font-size:.8rem}
  .badge{font-size:.72rem}
  .stTextInput > div > div > input{height:2.6rem;font-size:.95rem}
}

/* phones - extra top padding to prevent title clipping under Streamlit header */
@media (max-width: 600px){
  .block-container{padding-top:3.3rem;padding-left:.75rem;padding-right:.75rem}
  .hero-wrap{gap:.6rem}
  .hero-logo{width:40px;height:40px;padding:4px;border-radius:10px}
  .hero-logo svg{width:22px;height:22px}
  .hero-title{font-size:1.55rem;line-height:1.22}
  .hero-sub{font-size:.95rem;margin-bottom:1.1rem}
  .card{padding:.85rem;border-radius:14px}
  .chip{font-size:.75rem;padding:.22rem .5rem}
  .badge{font-size:.7rem;padding:.2rem .45rem}
  .stTextInput > div > div > input{height:2.4rem;font-size:.95rem}
  section.main .stColumns { flex-direction: column !important; gap: .75rem !important }
  .stButton > button, .stDownloadButton > button { width:100% !important }
  .plain-email textarea{height:260px}
}
</style>
""",
    unsafe_allow_html=True,
)

# --------------------- SIDEBAR ---------------------
with st.sidebar:
    st.header("⚙️ Settings")
    st.write("Paste a job post URL, choose your email tone and call-to-action, and generate a personalized cold email.")
    st.divider()

    TONES = ["Formal", "Friendly", "Confident", "Persuasive", "Enthusiastic"]
    CTAS = ["Request an Interview", "Ask for a Referral", "Offer to Discuss Further", "Request a Coffee Chat"]

    tone_choice = st.selectbox("🎭 Email Tone", TONES, index=TONES.index("Confident"))
    cta_choice = st.selectbox("📢 Call-to-Action", CTAS, index=CTAS.index("Request an Interview"))

    st.divider()
    st.caption("Quick Example URLs")
    PRESETS = {
        "Nike – SWE II": "https://careers.nike.com/software-engineer-ii-itc/job/R-71954",
        "Notion – Careers": "https://www.notion.so/careers",
        "Stripe – Engineering": "https://stripe.com/jobs/search?query=software",
    }
    preset_choice = st.selectbox("Presets", ["—"] + list(PRESETS.keys()))
    if st.button("Load Preset"):
        if preset_choice in PRESETS:
            st.session_state["preset_url"] = PRESETS[preset_choice]
            st.toast(f"Loaded preset: {preset_choice}", icon="✅")

# --------------------- HELPERS ---------------------
URL_RE = re.compile(r"^https?://", re.IGNORECASE)

def to_plain_text(s: str) -> str:
    """Convert potential markdown-ish output to plain text."""
    s = re.sub(r"```.*?```", "", s, flags=re.S)                 # fenced code blocks
    s = re.sub(r"`([^`]*)`", r"\1", s)                          # inline code
    s = re.sub(r"!\[.*?\]\(.*?\)", "", s)                       # images
    s = re.sub(r"\[(.*?)\]\((.*?)\)", r"\1", s)                 # links → text
    s = re.sub(r"[*_]{1,3}([^*_]+)[*_]{1,3}", r"\1", s)         # bold/italics
    s = re.sub(r"^\s{0,3}#{1,6}\s*", "", s, flags=re.M)         # headings
    s = re.sub(r"^\s{0,3}>\s*", "", s, flags=re.M)              # blockquotes
    s = re.sub(r"^\s*[-*_]{3,}\s*$", "", s, flags=re.M)         # horizontal rules
    s = re.sub(r"^\s*[-*•+]\s+", "", s, flags=re.M)             # bullets
    s = re.sub(r"\n{3,}", "\n\n", s)                            # collapse blanks
    return s.strip()

@st.cache_data(ttl=900, show_spinner=False)
def fetch_and_clean(url: str) -> str:
    doc = WebBaseLoader(url).load()[0]
    return clean_text(doc.page_content)

def normalize_skills(raw: Any) -> List[str]:
    if isinstance(raw, str):
        return [s.strip() for s in raw.split(",") if s.strip()]
    if isinstance(raw, list):
        return [str(s).strip() for s in raw if str(s).strip()]
    return [str(raw)]

def render_skill_chips(skills: List[str]):
    if not skills:
        st.caption("No skills parsed.")
        return
    chips = " ".join(f"<span class='chip'>{escape(s)}</span>" for s in skills)
    st.markdown(chips, unsafe_allow_html=True)

def download_name(prefix="email", ext="txt"):
    return f"{prefix}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.{ext}"

def render_plain_email(idx: int, text: str):
    """Show email in a readonly <textarea> with a working Copy button (no 'Copy code')."""
    st.markdown(
        f"""
        <div class="plain-email">
          <div class="email-toolbar">
            <button class="copy-btn" id="copy_btn_{idx}">Copy</button>
          </div>
          <textarea id="email_{idx}" readonly>{escape(text)}</textarea>
        </div>

        <script>
        const copyBtn{idx} = document.getElementById("copy_btn_{idx}");
        const textarea{idx} = document.getElementById("email_{idx}");
        if (copyBtn{idx} && textarea{idx}) {{
            copyBtn{idx}.addEventListener("click", async () => {{
                try {{
                    await navigator.clipboard.writeText(textarea{idx}.value);
                    const original = copyBtn{idx}.innerText;
                    copyBtn{idx}.innerText = "Copied!";
                    setTimeout(() => copyBtn{idx}.innerText = original, 1300);
                }} catch (err) {{
                    console.error("Copy failed:", err);
                }}
            }});
        }}
        </script>
        """,
        unsafe_allow_html=True,
    )


# --------------------- HERO ---------------------
st.markdown(
    """
    <div class="hero-wrap">
      <div class="hero-logo" aria-label="Email logo">
        <svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" role="img" aria-hidden="true">
          <path d="M20 4H4a2 2 0 0 0-2 2v12a2 2 0 0 0 2 2h16a2 2 0 0 0 2-2V6a2 2 0 0 0-2-2Zm0 2-8 5L4 6h16ZM4 18V8l8 5 8-5v10H4Z"/>
        </svg>
      </div>
      <div class="hero-title">Cold Email Generator</div>
    </div>
    <div class="hero-sub">Turn job posts into tailored, portfolio-backed cold emails instantly.</div>
    """,
    unsafe_allow_html=True,
)

col1, col2 = st.columns([3, 1])
with col1:
    default_url = st.session_state.get("preset_url", "https://careers.nike.com/software-engineer-ii-itc/job/R-71954")
    url_input = st.text_input("Job Post URL", value=default_url, placeholder="https://company.com/careers/role")
with col2:
    generate = st.button("🚀 Generate", type="primary", use_container_width=True)

# --------------------- MAIN ---------------------
if generate:
    if not url_input.strip() or not URL_RE.match(url_input.strip()):
        st.error("Please enter a valid `http(s)://` URL.")
    else:
        try:
            with st.spinner("🔎 Fetching & cleaning page…"):
                text = fetch_and_clean(url_input.strip())

            with st.spinner("🧠 Extracting job details…"):
                jobs = chain.extract_jobs(text)

            if not jobs:
                st.warning("No jobs detected. Try a specific job detail URL.")
            else:
                st.success(f"✅ Found {len(jobs)} job posting(s).")

            for i, job in enumerate(jobs, start=1):
                role = job.get("role", "Software Engineer")
                desc = job.get("description", "")
                exp = job.get("experience", "N/A")
                skills = normalize_skills(job.get("skills", []))

                # Header with badges
                st.markdown(
                    f"### #{i} — {escape(role)} "
                    f"<span class='badge'>{escape(tone_choice)}</span>"
                    f"<span class='badge'>{escape(cta_choice)}</span>",
                    unsafe_allow_html=True,
                )

                # Job summary card
                st.markdown("<div class='card'>", unsafe_allow_html=True)
                st.subheader("Summary", anchor=False)
                st.write(desc)
                st.markdown(f"**Experience:** {escape(str(exp))}")
                st.subheader("Skills", anchor=False)
                render_skill_chips(skills)
                st.markdown("</div>", unsafe_allow_html=True)

                # Ask LLM to write (links intentionally empty; email should be plain text)
                job_with_prefs = {**job, "tone": tone_choice, "cta": cta_choice}
                with st.spinner("✍️ Writing tailored email…"):
                    email_md = chain.write_mail(job_with_prefs, [])

                # Plain text conversion + UI
                email_txt = to_plain_text(email_md)
                render_plain_email(i, email_txt)

                # Download .txt
                st.download_button(
                    label="⬇️ Download Email (.txt)",
                    data=email_txt,
                    file_name=download_name(ext="txt"),
                    mime="text/plain",
                    use_container_width=True,
                    key=f"download_email_{i}",
                )

                st.markdown("<hr />", unsafe_allow_html=True)

        except Exception as e:
            msg = str(e).lower()
            if "rate limit" in msg or "429" in msg:
                st.error(
                    "🚦 Groq rate limit reached for your org. Please try again later, "
                    "or switch to a smaller model / separate API key."
                )
            else:
                st.error(f"⚠️ An error occurred: {e}")
