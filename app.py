"""
Milestone 4 – NurseAssist: Generative AI Clinical Companion
DSC 670 Term Project | Bellevue University

A Streamlit application that surfaces the fine-tuned nursing AI assistant
as a practical, nurse-centered tool for use during a clinical shift.

Fine-tuned model: ft:gpt-3.5-turbo-0125:personal:nurseassist-v1:DdpNvVC2

To run this app:
    streamlit run app.py

Requirements:
    pip install streamlit openai python-dotenv

Environment variables (.env file or system environment):
    OPENAI_API_KEY   — your OpenAI API key
    FINE_TUNED_MODEL — optional override (defaults to the trained model below)
"""

import streamlit as st
import openai
import os
import json
from datetime import datetime
from dotenv import load_dotenv

# ─────────────────────────────────────────────────────────
# Configuration
# ─────────────────────────────────────────────────────────

load_dotenv()

OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY", "")
FINE_TUNED_MODEL = os.environ.get(
    "FINE_TUNED_MODEL",
    "ft:gpt-3.5-turbo-0125:personal:nurseassist-v1:DdpNvVC2"
)

if OPENAI_API_KEY:
    openai.api_key = OPENAI_API_KEY

SYSTEM_PROMPT = (
    "You are NurseAssist, an AI clinical companion for registered nurses working in acute care settings. "
    "You provide evidence-based medication information, help draft clinical documentation, answer nursing "
    "practice questions, and support patient education. Always prioritize patient safety. When uncertain, "
    "recommend the nurse consult the attending physician or pharmacist. You do not replace clinical judgment. "
    "Keep responses concise and clinically focused. Use structured formatting (numbered lists, bold headers) "
    "when it improves clarity."
)

QUICK_PROMPTS = {
    "💊  Medication Interaction Check": (
        "My patient is currently taking [Drug A] and was just ordered [Drug B]. "
        "Are there any interactions I should be aware of before administering?"
    ),
    "📋  Generate SBAR Handoff Note": (
        "Write an SBAR handoff note for a [age]-year-old [male/female] patient admitted for [diagnosis]. "
        "Current vitals: [vitals]. Recent interventions: [interventions]."
    ),
    "👩‍⚕️  Patient Education Script": (
        "Write a simple, plain-language explanation I can use to educate my patient about [medication/procedure/diagnosis]."
    ),
    "🔍  Clinical Decision Support": (
        "My patient has [symptom/lab value/clinical finding]. "
        "What should I be thinking about and what are my next steps?"
    ),
    "⚖️  Verify Medication Dosing": (
        "My patient weighs [weight] kg. The order is for [drug] [dose] [route] [frequency]. "
        "Is this within normal range for an adult?"
    ),
    "📊  Interpret Lab Value": (
        "My patient's [lab name] is [value]. Is this within normal range, and what are the clinical implications?"
    ),
}

# ─────────────────────────────────────────────────────────
# Page Config
# ─────────────────────────────────────────────────────────

st.set_page_config(
    page_title="NurseAssist – AI Clinical Companion",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────────────────
# Custom CSS
# ─────────────────────────────────────────────────────────

st.markdown("""
<style>
    /* ── Global ── */
    html, body, [class*="css"] { font-family: "Inter", "Segoe UI", sans-serif; }
    .main { background-color: #F4F7FB; }

    /* ── App header ── */
    .nurse-header {
        background: linear-gradient(135deg, #0D3B5E 0%, #1B6CA8 100%);
        color: white;
        padding: 1.4rem 1.8rem;
        border-radius: 12px;
        margin-bottom: 1.2rem;
        display: flex;
        align-items: center;
        gap: 1rem;
    }
    .nurse-header h1 { margin: 0; font-size: 1.65rem; font-weight: 700; letter-spacing: -0.02em; }
    .nurse-header p  { margin: 0.25rem 0 0 0; font-size: 0.875rem; opacity: 0.82; }

    /* ── Chat bubbles ── */
    .user-msg {
        background: #E8F2FC;
        border-left: 4px solid #1B6CA8;
        padding: 0.85rem 1.1rem;
        border-radius: 8px;
        margin: 0.55rem 0;
    }
    .assistant-msg {
        background: #FFFFFF;
        border-left: 4px solid #0FA37F;
        padding: 0.85rem 1.1rem;
        border-radius: 8px;
        margin: 0.55rem 0;
        box-shadow: 0 2px 6px rgba(0,0,0,0.06);
    }
    .msg-label {
        font-size: 0.72rem;
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: 0.07em;
        margin-bottom: 0.45rem;
    }
    .user-label      { color: #1B6CA8; }
    .ai-label        { color: #0A7A5E; }

    /* ── Safety banner ── */
    .safety-banner {
        background: #FFFBEA;
        border: 1px solid #F5C842;
        border-radius: 8px;
        padding: 0.65rem 1rem;
        font-size: 0.81rem;
        color: #7A5C00;
        margin-bottom: 1rem;
    }

    /* ── Sidebar quick-prompt buttons ── */
    .stButton > button {
        background: #F0F6FC;
        border: 1px solid #C5D9EF;
        color: #1B5E8B;
        font-size: 0.8rem;
        border-radius: 7px;
        padding: 0.45rem 0.75rem;
        width: 100%;
        text-align: left;
        transition: background 0.15s;
    }
    .stButton > button:hover {
        background: #D6EAFF;
        border-color: #1B6CA8;
    }

    /* ── Metric cards ── */
    [data-testid="metric-container"] {
        background: white;
        border-radius: 10px;
        padding: 0.8rem 1rem;
        box-shadow: 0 1px 4px rgba(0,0,0,0.07);
    }

    /* ── Sidebar section labels ── */
    .sidebar-label {
        font-size: 0.72rem;
        text-transform: uppercase;
        letter-spacing: 0.07em;
        color: #94A3B8;
        margin-bottom: 0.4rem;
        margin-top: 0.1rem;
    }

    /* ── Tab bar ── */
    .stTabs [data-baseweb="tab"] { font-size: 0.85rem; }
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────
# Session-State Defaults
# ─────────────────────────────────────────────────────────

for key, default in [
    ("messages", []),
    ("audit_log", []),
    ("api_key_confirmed", bool(OPENAI_API_KEY)),
    ("total_queries", 0),
]:
    if key not in st.session_state:
        st.session_state[key] = default

# ─────────────────────────────────────────────────────────
# Sidebar
# ─────────────────────────────────────────────────────────

with st.sidebar:
    st.markdown("## 🏥 NurseAssist")
    st.caption("AI Clinical Companion · Prototype v1.0")
    st.divider()

    # API Key
    st.markdown('<p class="sidebar-label">API Configuration</p>', unsafe_allow_html=True)
    if st.session_state.api_key_confirmed:
        st.success("✅ API key loaded", icon=None)
    else:
        key_in = st.text_input("OpenAI API Key", type="password",
                               help="Enter key or add OPENAI_API_KEY to .env")
        if key_in:
            openai.api_key = key_in
            st.session_state.api_key_confirmed = True
            st.rerun()

    short_model = FINE_TUNED_MODEL.split(":")[-1] if ":" in FINE_TUNED_MODEL else FINE_TUNED_MODEL
    st.caption(f"Model: `{short_model}`")
    st.divider()

    # Quick Prompts
    st.markdown('<p class="sidebar-label">Quick Prompt Templates</p>', unsafe_allow_html=True)
    st.caption("Click any template to pre-fill the input.")
    for label, template in QUICK_PROMPTS.items():
        if st.button(label, key=f"qp_{label}"):
            st.session_state["prefill"] = template

    st.divider()

    # Session controls
    st.markdown('<p class="sidebar-label">Session Controls</p>', unsafe_allow_html=True)
    if st.button("🗑️  Clear Conversation"):
        st.session_state.messages = []
        st.rerun()

    if st.button("📥  Export Audit Log"):
        if st.session_state.audit_log:
            log_json = json.dumps(st.session_state.audit_log, indent=2)
            st.download_button(
                "Download JSON",
                data=log_json,
                file_name=f"nurseassist_audit_{datetime.now().strftime('%Y%m%d_%H%M')}.json",
                mime="application/json",
            )
        else:
            st.caption("No entries yet.")

    st.divider()
    st.caption(
        "⚠️ Educational prototype only. All AI responses must be verified by a "
        "licensed clinician before any clinical action."
    )

# ─────────────────────────────────────────────────────────
# Main Panel Header
# ─────────────────────────────────────────────────────────

st.markdown("""
<div class="nurse-header">
    <div>
        <h1>🏥 NurseAssist – AI Clinical Companion</h1>
        <p>Evidence-based support for medication safety, clinical documentation &amp; patient education</p>
    </div>
</div>
""", unsafe_allow_html=True)

st.markdown("""
<div class="safety-banner">
    ⚠️ <strong>Clinical Disclaimer:</strong> NurseAssist is an AI prototype designed to <em>assist</em>,
    not replace, registered nurse judgment. Always verify AI-generated information with your facility's
    policies, pharmacist, or attending physician before acting. Not validated for direct patient care.
</div>
""", unsafe_allow_html=True)

if not st.session_state.api_key_confirmed:
    st.warning("No API key found. Enter it in the sidebar or add `OPENAI_API_KEY` to your `.env` file.")
    st.stop()

# ─────────────────────────────────────────────────────────
# Tabs: Chat | About
# ─────────────────────────────────────────────────────────

tab_chat, tab_about = st.tabs(["💬 Chat", "ℹ️ About"])

# ───── About Tab ─────
with tab_about:
    st.markdown("### About NurseAssist")
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("""
**What it does**

NurseAssist is a fine-tuned GPT-3.5 model trained on nursing-specific clinical scenarios. It helps registered nurses with:

- **Medication safety** – interaction checks, dosing verification
- **Documentation** – SBAR handoff note generation
- **Clinical Q&A** – lab interpretation, post-op monitoring thresholds
- **Patient education** – plain-language explanations

**Model details**

| Field | Value |
|---|---|
| Base model | GPT-3.5-Turbo-0125 |
| Fine-tune job | nurseassist-v1 |
| Training examples | 200+ nursing scenarios |
| Evaluation | 20-case held-out test set |
""")
    with col2:
        st.markdown("""
**Use cases demonstrated in training**

1. Medication interaction identification (e.g. warfarin + metronidazole)
2. Weight-based and absolute dosing verification
3. SBAR note generation from structured vitals
4. Post-operative oliguria evaluation
5. Patient-friendly medication explanations

**Limitations**

- Prototype only — not FDA-cleared or institutionally validated
- Should never be the sole basis for a clinical decision
- Does not access live EHR data or real-time drug databases
- Responses may occasionally be incomplete or outdated

**Project context**

DSC 670 — Term Project, Bellevue University
""")

# ───── Chat Tab ─────
with tab_chat:

    # ── Chat History ──
    chat_box = st.container()
    with chat_box:
        if not st.session_state.messages:
            st.info(
                "👋 Welcome! Type a clinical question below, or choose a **Quick Prompt** "
                "from the sidebar to load a template."
            )
        else:
            for msg in st.session_state.messages:
                if msg["role"] == "user":
                    st.markdown(
                        f'<div class="user-msg">'
                        f'<div class="msg-label user-label">🧑‍⚕️ You (Nurse)</div>'
                        f'{msg["content"]}'
                        f'</div>',
                        unsafe_allow_html=True,
                    )
                elif msg["role"] == "assistant":
                    content_html = msg["content"].replace("\n", "<br>")
                    st.markdown(
                        f'<div class="assistant-msg">'
                        f'<div class="msg-label ai-label">🤖 NurseAssist</div>'
                        f'{content_html}'
                        f'</div>',
                        unsafe_allow_html=True,
                    )

    st.divider()

    # ── Input Area ──
    prefill_val = st.session_state.pop("prefill", "")
    col_input, col_btn = st.columns([5, 1])

    with col_input:
        user_input = st.text_area(
            "Ask NurseAssist...",
            value=prefill_val,
            placeholder="e.g. 'My patient is on digoxin and amiodarone was just ordered. What should I check?'",
            height=110,
            label_visibility="collapsed",
            key="user_input_box",
        )

    with col_btn:
        st.markdown("<br>", unsafe_allow_html=True)
        send = st.button("Send ➤", use_container_width=True, type="primary")
        st.button("Clear", use_container_width=True)

    # ── Message Handling ──
    def get_ai_response(user_message: str) -> str:
        """Send the current message (plus recent history) to the fine-tuned model."""
        payload = [{"role": "system", "content": SYSTEM_PROMPT}]
        # Keep last 12 turns for context (6 exchanges)
        for m in st.session_state.messages[-12:]:
            payload.append({"role": m["role"], "content": m["content"]})
        payload.append({"role": "user", "content": user_message})

        resp = openai.chat.completions.create(
            model=FINE_TUNED_MODEL,
            messages=payload,
            max_tokens=700,
            temperature=0.3,
        )
        return resp.choices[0].message.content

    if send and user_input.strip():
        st.session_state.messages.append({"role": "user", "content": user_input.strip()})

        with st.spinner("NurseAssist is thinking…"):
            try:
                reply = get_ai_response(user_input.strip())
                st.session_state.messages.append({"role": "assistant", "content": reply})
                st.session_state.total_queries += 1
                st.session_state.audit_log.append({
                    "timestamp": datetime.now().isoformat(),
                    "model": FINE_TUNED_MODEL,
                    "user_query": user_input.strip(),
                    "ai_response": reply,
                })
            except openai.AuthenticationError:
                st.error("Authentication error — check your OPENAI_API_KEY.")
            except openai.RateLimitError:
                st.error("Rate limit reached. Please wait a moment and try again.")
            except Exception as exc:
                st.error(f"Error: {exc}")

        st.rerun()

    # ── Session Metrics ──
    if st.session_state.messages:
        st.divider()
        n_queries = len([m for m in st.session_state.messages if m["role"] == "user"])
        c1, c2, c3 = st.columns(3)
        c1.metric("Queries This Session", n_queries)
        c2.metric("Audit Log Entries", len(st.session_state.audit_log))
        c3.metric("Active Model", short_model)
