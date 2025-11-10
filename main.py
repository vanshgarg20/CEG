# main.py
import math
import re
from html import escape
from datetime import datetime
from typing import Any, List

import streamlit as st
import streamlit.components.v1 as components
from langchain_community.document_loaders import WebBaseLoader

from html import escape
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

# --------------------- STYLES (base + responsive) ---------------------
st.markdown(
    """
<style>
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

/* ====================== EMAIL VIEWER ====================== */
.plain-email{ margin-top:.4rem; margin-bottom:.25rem; }  /* ultra tight bottom margin */
.email-toolbar{ display:flex; justify-content:flex-end; gap:.5rem; margin-bottom:.35rem; }
.copy-btn{
  border:1px solid rgba(255,255,255,.15);
  background:rgba(255,255,255,.08);
  padding:.35rem .7rem; border-radius:8px; cursor:pointer; color:inherit;
}
.copy-btn:active{ transform:translateY(1px) }
.emailbox{
  background:rgba(255,255,255,.03);
  border:1px solid rgba(255,255,255,.15);
  border-radius:12px;
  padding:14px;
  font:0.92rem/1.5 ui-monospace, SFMono-Regular, Menlo, Consolas, "Liberation Mono", monospace;
  white-space:pre-wrap; word-break:break-word; overflow-wrap:anywhere;
}
.hidden-copy{ position:absolute; left:-9999px; top:-9999px; height:0; width:0; opacity:0; }

/* Tight spacing before Download button */
div[data-testid="stDownloadButton"]{
  margin-top:3px !important;      /* tighter gap */
  margin-bottom:0 !important;
  padding-top:0 !important;
}

/* ====================== RESPONSIVE STYLES ====================== */
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

  /* ultra-tight for mobile */
  .plain-email{ margin-bottom:.22rem; }
  div[data-testid="stDownloadButton"]{ margin-top:2px !important; }
  .emailbox{ font-size:.98rem; line-height:1.5; padding:15px 13px; }
}
</style>
""",
    unsafe_allow_html=True,
)

# ---- THEME COLORS (fallbacks) ----
BASE = (st.get_option("theme.base") or "dark").lower()
THEME_BG  = st.get_option("theme.backgroundColor") or ("#0E1117" if BASE == "dark" else "#FFFFFF")
THEME_TX  = st.get_option("theme.textColor")       or ("#FAFAFA" if BASE == "dark" else "#0B0B0B")
THEME_SEC = st.get_option("theme.secondaryBackgroundColor") or ("#262730" if BASE == "dark" else "#F0F2F6")
BORDER_RG = "rgba(255,255,255,.15)" if BASE == "dark" else "rgba(0,0,0,.12)"
BTN_BG    = "rgba(255,255,255,.08)" if BASE == "dark" else "rgba(0,0,0,.05)"
AREA_BG   = "rgba(255,255,255,.03)" if BASE == "dark" else "rgba(0,0,0,.02)"

# ---- Email box knobs (customize here) ----
EMAIL_BOX_DESKTOP_HEIGHT = 300   # Medium box height on desktop
EMAIL_BOX_RADIUS         = 12    # Border radius of the email container
EMAIL_FONT_SIZE_DESKTOP  = "0.92rem"
EMAIL_FONT_SIZE_MOBILE   = "0.98rem"
EMAIL_LINE_HEIGHT        = "1.5" # Line-height both views


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
    s = re.sub(r"```.*?```", "", s, flags=re.S)
    s = re.sub(r"`([^`]*)`", r"\1", s)
    s = re.sub(r"!\[.*?\]\(.*?\)", "", s)
    s = re.sub(r"\[(.*?)\]\((.*?)\)", r"\1", s)
    s = re.sub(r"[*_]{1,3}([^*_]+)[*_]{1,3}", r"\1", s)
    s = re.sub(r"^\s{0,3}#{1,6}\s*", "", s, flags=re.M)
    s = re.sub(r"^\s{0,3}>\s*", "", s, flags=re.M)
    s = re.sub(r"^\s*[-*_]{3,}\s*$", "", s, flags=re.M)
    s = re.sub(r"^\s*[-*•+]\s+", "", s, flags=re.M)
    s = re.sub(r"\n{3,}", "\n\n", s)
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

import streamlit.components.v1 as components
from html import escape

# ---- (optional) tweakable knobs at top of your file ----
# ---- Optional knobs (upar rakh lo) ----
EMAIL_BOX_DESKTOP_HEIGHT = 300   # medium desktop look (min visual height)
EMAIL_FONT_SIZE_DESKTOP  = "0.92rem"
EMAIL_FONT_SIZE_MOBILE   = "0.98rem"
EMAIL_LINE_HEIGHT        = 1.5   # CSS line-height
EMAIL_LINE_PX_DESKTOP    = 22    # px approx (estimate only)
EMAIL_LINE_PX_MOBILE     = 26    # px approx (estimate only)

import math
import streamlit.components.v1 as components
from html import escape

def _estimate_iframe_height_for(text: str) -> int:
    # Rough lines on desktop & mobile, pick the larger to avoid cut on phone
    wrap_d = 90   # ~chars per line desktop
    wrap_m = 38   # ~chars per line narrow phones
    lines_d = 0
    lines_m = 0
    for raw in text.splitlines() or [""]:
        L = max(1, len(raw))
        lines_d += max(1, math.ceil(L / wrap_d))
        lines_m += max(1, math.ceil(L / wrap_m))
    est_d = 140 + lines_d * EMAIL_LINE_PX_DESKTOP   # padding + lines
    est_m = 160 + lines_m * EMAIL_LINE_PX_MOBILE    # padding + lines
    est   = max(est_d, est_m)
    # Add a tiny bottom buffer so first paint pe bhi overlap na ho
    est  += 40
    # Clamp: at least desktop min, at most 4000
    return max(EMAIL_BOX_DESKTOP_HEIGHT + 80, min(est, 4000))

def render_plain_email(idx: int, text: str):
    """
    FINAL: No inner scroll, no overlap, no extra gap.
    - Desktop: medium look (min height), iframe auto-resizes to exact content.
    - Mobile: full content (same lines/words), iframe auto-resizes (no cut).
    - Copy button works; theme colors used.
    """
    initial_height = _estimate_iframe_height_for(text)

    template = """<!doctype html>
<html>
  <head>
    <meta charset="utf-8"/>
    <meta name="viewport" content="width=device-width,initial-scale=1"/>
    <style>
      :root{
        --bg: %%BG%%; --tx: %%TX%%; --border: %%BORDER%%; --btnbg: %%BTN_BG%%; --areabg: %%AREA_BG%%;
        --radius: 12px; --fs-d: %%FS_DESKTOP%%; --fs-m: %%FS_MOBILE%%; --lh: %%LH%%;
        --h-desk: %%H_DESKTOP%%px;
      }
      html,body{
        margin:0; padding:0; background:var(--bg); color:var(--tx);
        font-family:system-ui,-apple-system,Segoe UI,Roboto,Ubuntu,Cantarell,'Helvetica Neue',Arial,'Noto Sans',sans-serif;
        overflow:hidden; /* iframe ke andar scroll nahi chahiye */
      }
      .wrap{ width:100%; }
      .toolbar{ display:flex; justify-content:flex-end; gap:.5rem; margin:0 0 8px 0; }
      .btn{
        border:1px solid var(--border); background:var(--btnbg); color:var(--tx);
        padding:.35rem .7rem; border-radius:8px; cursor:pointer; font-size:.9rem;
      }
      .btn:active{ transform:translateY(1px); }

      /* Non-focus email block (no focus ring) */
      .emailbox{
        background:var(--areabg);
        border:1px solid var(--border);
        border-radius:var(--radius);
        padding:14px;
        font: var(--fs-d)/var(--lh) ui-monospace, SFMono-Regular, Menlo, Consolas, 'Liberation Mono', monospace;
        white-space:pre-wrap; word-break:break-word; overflow-wrap:anywhere;
        margin:0; /* no big extra margins */
      }
      .emailbox:focus, .emailbox:focus-visible{ outline:0; box-shadow:none; }

      /* Desktop: give a visual floor height (looks like medium box) */
      @media (min-width:601px){
        .emailbox{ min-height:var(--h-desk); }
      }
      /* Mobile: comfortable font; natural height */
      @media (max-width:600px){
        .emailbox{ font-size:var(--fs-m); line-height:var(--lh); padding:16px 14px; }
      }

      /* Small spacer so Download button has breathing room (not huge) */
      .spacer{ height: 12px; }
      /* Hidden textarea for copy preserving linebreaks */
      textarea.hidden-copy{ position:absolute; left:-9999px; top:-9999px; height:0; width:0; opacity:0; }
    </style>
  </head>
  <body>
    <div class="wrap">
      <div class="toolbar">
        <button id="copy_btn_%%IDX%%" class="btn">Copy</button>
      </div>
      <pre id="email_view_%%IDX%%" class="emailbox">%%TEXT%%</pre>
      <div class="spacer"></div>
      <textarea id="copy_src_%%IDX%%" class="hidden-copy" readonly>%%TEXT%%</textarea>
    </div>

    <script>
      (function(){
        const btn = document.getElementById('copy_btn_%%IDX%%');
        const src = document.getElementById('copy_src_%%IDX%%');
        if (btn && src){
          btn.addEventListener('click', async () => {
            try{
              src.focus(); src.select();
              const ok = document.execCommand('copy');
              if (!ok && navigator.clipboard) await navigator.clipboard.writeText(src.value);
              const old = btn.innerText; btn.innerText = 'Copied!';
              setTimeout(()=>btn.innerText = old, 1100);
            }catch(e){ console.error('Copy failed', e); }
          });
        }

        // Exact fit — measure wrapper's real height and set iframe to it.
        function fit(){
          try{
            const wrap = document.querySelector('.wrap');
            const h = Math.ceil(wrap.getBoundingClientRect().height) + 8; // tiny padding
            if (window.frameElement) window.frameElement.style.height = h + 'px';
            document.documentElement.style.height = h + 'px';
          }catch(_){}
        }
        // Run multiple times to defeat late font/layout reflows
        if (window.ResizeObserver){
          const ro = new ResizeObserver(fit);
          ro.observe(document.body);
          ro.observe(document.querySelector('.wrap'));
        }
        window.addEventListener('load', fit);
        window.addEventListener('resize', fit);
        if (document.fonts && document.fonts.ready) document.fonts.ready.then(fit);
        setTimeout(fit, 50); setTimeout(fit, 250); setTimeout(fit, 800);
      })();
    </script>
  </body>
</html>"""

    html = (
        template
        .replace("%%IDX%%", str(idx))
        .replace("%%TEXT%%", escape(text))
        .replace("%%BG%%", THEME_BG).replace("%%TX%%", THEME_TX)
        .replace("%%BORDER%%", BORDER_RG).replace("%%BTN_BG%%", BTN_BG)
        .replace("%%AREA_BG%%", AREA_BG)
        .replace("%%FS_DESKTOP%%", EMAIL_FONT_SIZE_DESKTOP)
        .replace("%%FS_MOBILE%%", EMAIL_FONT_SIZE_MOBILE)
        .replace("%%LH%%", str(EMAIL_LINE_HEIGHT))
        .replace("%%H_DESKTOP%%", str(EMAIL_BOX_DESKTOP_HEIGHT))
    )

    # Initial safe height (mobile + desktop estimate). JS will tighten exactly.
    components.html(html, height=initial_height, scrolling=False)


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

                # Ask LLM to write (plain text)
                job_with_prefs = {**job, "tone": tone_choice, "cta": cta_choice}
                with st.spinner("✍️ Writing tailored email…"):
                    email_md = chain.write_mail(job_with_prefs, [])

                email_txt = to_plain_text(email_md)
                render_plain_email(i, email_txt)

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
