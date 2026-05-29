# ============================================================
# app.py — Urdu POS Tagger —
# ============================================================

import streamlit as st
import numpy as np
import joblib
import pandas as pd

st.set_page_config(page_title="Urdu POS Tagger", page_icon="🔤", layout="wide")

st.markdown("""
<style>
    .stApp { background: linear-gradient(135deg, #0f0c29, #302b63, #24243e); color: white; }
    #MainMenu, footer, header { visibility: hidden; }
    .main-title {
        text-align: center; font-size: 3rem; font-weight: 800;
        background: linear-gradient(90deg, #a78bfa, #60a5fa, #34d399);
        -webkit-background-clip: text; -webkit-text-fill-color: transparent;
        padding: 20px 0 5px 0;
    }
    .sub-title { text-align: center; color: #94a3b8; font-size: 1rem; margin-bottom: 30px; }
    .stTextArea textarea {
        background-color: #f0eeff !important;
        border: 1.5px solid rgba(167,139,250,0.5) !important;
        border-radius: 12px !important;
        color: #1a1a2e !important;
        font-size: 1.1rem !important;
        direction: rtl !important;
        text-align: right !important;
        unicode-bidi: bidi-override !important;
    }
    .stButton > button {
        background: linear-gradient(135deg, #a78bfa, #60a5fa) !important;
        color: white !important; border: none !important;
        border-radius: 10px !important; font-weight: 600 !important;
        transition: all 0.3s ease !important;
    }
    .stButton > button:hover {
        transform: translateY(-2px) !important;
        box-shadow: 0 8px 20px rgba(167,139,250,0.4) !important;
    }
    .stat-card {
        background: rgba(255,255,255,0.05); border: 1px solid rgba(255,255,255,0.1);
        border-radius: 16px; padding: 20px; text-align: center;
    }
    .stat-number {
        font-size: 2rem; font-weight: 800;
        background: linear-gradient(90deg, #a78bfa, #60a5fa);
        -webkit-background-clip: text; -webkit-text-fill-color: transparent;
    }
    .stat-label { color: #94a3b8; font-size: 0.85rem; margin-top: 4px; }
    .output-box {
        background: rgba(255,255,255,0.03); border: 1px solid rgba(255,255,255,0.08);
        border-radius: 16px; padding: 20px; min-height: 100px;
    }
    .section-header {
        color: #a78bfa; font-size: 1rem; font-weight: 700;
        text-transform: uppercase; letter-spacing: 1px; margin: 20px 0 10px 0;
    }
    .legend-item { display: flex; align-items: center; gap: 8px; margin: 6px 0; font-size: 0.9rem; color: #cbd5e1; }
    .legend-dot { width: 12px; height: 12px; border-radius: 50%; display: inline-block; flex-shrink: 0; }
    hr { border-color: rgba(255,255,255,0.1) !important; }
    div[data-testid="stForm"] { border: none !important; padding: 0 !important; }
</style>
""", unsafe_allow_html=True)

TAG_COLORS = {
    'NOUN':'#4A90D9','VERB':'#E74C3C','ADJ':'#2ECC71','ADP':'#9B59B6',
    'ADV':'#F39C12','PRON':'#1ABC9C','PROPN':'#E67E22','AUX':'#3498DB',
    'CCONJ':'#95A5A6','SCONJ':'#7F8C8D','DET':'#D35400','NUM':'#27AE60',
    'PART':'#8E44AD','PUNCT':'#BDC3C7','INTJ':'#E91E63','X':'#999999'
}

@st.cache_resource
def load_model():
    lr    = joblib.load('models/lr_model.pkl')
    tfidf = joblib.load('models/tfidf.pkl')
    return lr, tfidf

def predict(text, lr, tfidf):
    words    = text.strip().split()
    features = tfidf.transform(words)
    tags     = lr.predict(features)
    probs    = lr.predict_proba(features)
    confs    = [float(np.max(p)) for p in probs]
    return words, list(tags), confs

def build_html(words, tags, confs):
    html = "<div style='font-size:1.1rem;line-height:4;direction:rtl;text-align:right;padding:10px;'>"
    for word, tag, conf in zip(words, tags, confs):
        color = TAG_COLORS.get(tag, '#999999')
        html += (f"<span style='background:{color};color:white;padding:8px 14px;"
                 f"margin:5px;border-radius:10px;display:inline-block;"
                 f"box-shadow:0 4px 12px rgba(0,0,0,0.3);'>"
                 f"{word}<br><small style='opacity:0.9'><b>{tag}</b> {conf*100:.0f}%</small></span> ")
    html += "</div>"
    return html

# ── Session state ──
if "typed_text" not in st.session_state:
    st.session_state.typed_text = ""

# ── HEADER ──
st.markdown('<div class="main-title">🔤 Urdu POS Tagger</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-title">Part-of-Speech Tagging for Urdu Text using Machine Learning</div>', unsafe_allow_html=True)

# ── STATS ──
c1, c2, c3, c4 = st.columns(4)
for col, num, label in zip([c1,c2,c3,c4],
  ["96.4%", "16K+", "16", "5"],
  ["Best Accuracy", "Training Sentences", "POS Tag Types", "Models Trained"]):
    col.markdown(f'<div class="stat-card"><div class="stat-number">{num}</div>'
                 f'<div class="stat-label">{label}</div></div>', unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

# ── MAIN LAYOUT ──
left, right = st.columns([2, 1])

with left:
    st.markdown('<div class="section-header">✏️ Enter Urdu Sentence</div>', unsafe_allow_html=True)

    # ── Example buttons ──
    st.markdown('<div style="color:#94a3b8;font-size:0.85rem;margin-bottom:8px">💡 Click an example:</div>',
                unsafe_allow_html=True)

    examples = ["میں اسکول جاتا ہوں", "پاکستان ایک خوبصورت ملک ہے", "آج موسم بہت اچھا ہے"]
    ex1, ex2, ex3 = st.columns(3)

    if ex1.button(examples[0], key="ex1", use_container_width=True):
        st.session_state.typed_text = examples[0]
        st.rerun()
    if ex2.button(examples[1], key="ex2", use_container_width=True):
        st.session_state.typed_text = examples[1]
        st.rerun()
    if ex3.button(examples[2], key="ex3", use_container_width=True):
        st.session_state.typed_text = examples[2]
        st.rerun()

    # ── Text area ──
    input_text = st.text_area(
        label="Urdu Input",
        value=st.session_state.typed_text,
        placeholder="یہاں اردو جملہ لکھیں — Type your Urdu sentence here...",
        height=120,
        label_visibility="hidden",
    )


    # ── Action buttons ──
    b1, b2 = st.columns([3, 1])
    predict_btn = b1.button("🔍  Tag This Sentence", type="primary", use_container_width=True)
    clear_btn = b2.button("🗑️ Clear", key="clear_btn", use_container_width=True)

    if clear_btn:
        st.session_state.typed_text = ""
        st.rerun()

with right:
    st.markdown('<div class="section-header">🎨 POS Tag Legend</div>', unsafe_allow_html=True)
    for color, tag, urdu in [
        ('#4A90D9','NOUN','اسم'),('#E74C3C','VERB','فعل'),
        ('#2ECC71','ADJ','صفت'),('#9B59B6','ADP','حرف جار'),
        ('#F39C12','ADV','متعلق فعل'),('#1ABC9C','PRON','ضمیر'),
        ('#E67E22','PROPN','علم'),('#3498DB','AUX','فعل معاون'),
        ('#27AE60','NUM','عدد'),('#BDC3C7','PUNCT','رموز'),
    ]:
        st.markdown(f'<div class="legend-item">'
                    f'<span class="legend-dot" style="background:{color}"></span>'
                    f'<b style="color:white">{tag}</b>'
                    f'<span style="color:#64748b"> — {urdu}</span></div>',
                    unsafe_allow_html=True)

# ── OUTPUT ──
final_text = input_text if input_text and input_text.strip() else st.session_state.typed_text

if predict_btn and final_text and final_text.strip():
    with st.spinner("🔄 Analyzing..."):
        lr, tfidf = load_model()
        words, tags, confs = predict(final_text, lr, tfidf)

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown('<div class="section-header">🎯 Tagged Output</div>', unsafe_allow_html=True)
    st.markdown(f'<div class="output-box">{build_html(words, tags, confs)}</div>',
                unsafe_allow_html=True)
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown('<div class="section-header">📋 Token Details</div>', unsafe_allow_html=True)
    st.dataframe(pd.DataFrame({
        "Token": words, "POS Tag": tags,
        "Confidence": [f"{c*100:.1f}%" for c in confs]
    }), use_container_width=True, hide_index=True)

elif predict_btn:
    st.warning("⚠️ Please enter an Urdu sentence first!")

# ── MODEL COMPARISON ──
st.markdown("<br><hr>", unsafe_allow_html=True)
st.markdown('<div class="section-header">📊 Model Performance Comparison</div>', unsafe_allow_html=True)

m1, m2, m3 = st.columns(3)
for col, name, acc, f1, color in zip([m1,m2,m3],
    ["Logistic Regression","BiLSTM","XLM-RoBERTa"],
    ["87.18%","96.28%","~96.5%"],
    ["75.42%","85.02%","85.80%"],
    ["#64748b","#a78bfa","#34d399"]):
    col.markdown(
        f'<div class="stat-card">'
        f'<div style="color:{color};font-weight:700;margin-bottom:10px">{name}</div>'
        f'<div style="color:white;font-size:1.4rem;font-weight:800">{acc}</div>'
        f'<div style="color:#94a3b8;font-size:0.8rem">Accuracy</div>'
        f'<div style="color:white;font-size:1.1rem;font-weight:700;margin-top:8px">{f1}</div>'
        f'<div style="color:#94a3b8;font-size:0.8rem">F1 Macro</div>'
        f'</div>', unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)
st.markdown('<div style="text-align:center;color:#475569;font-size:0.8rem">'
            'Fareeha — Urdu POS Tagger — NLP Project</div>', unsafe_allow_html=True)