
import io
import json
import sqlite3
from datetime import datetime
from pathlib import Path
from urllib.parse import urlencode

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import qrcode
import streamlit as st
from sklearn.ensemble import RandomForestRegressor
from streamlit_autorefresh import st_autorefresh

APP_TITLE = "Live Math Grade Predictor"
DB_PATH = Path("audience_state.db")
SEED = 42
DEFAULT_PUBLIC_URL = "https://afrazmathapp.streamlit.app"

st.set_page_config(page_title=APP_TITLE, page_icon="📚", layout="wide")

st.markdown("""
<style>
.stApp, [data-testid="stAppViewContainer"] {
    background: radial-gradient(circle at top, #14305e 0%, #0a1630 35%, #050b16 100%);
    color: #f8fafc;
}
[data-testid="stHeader"] { background: rgba(0,0,0,0); }
.block-container { max-width: 1380px; padding-top: 1rem; padding-bottom: 1rem; }
h1,h2,h3,h4,h5,h6,p,label,div,span { color: #f8fafc; }
.hero {
    background: linear-gradient(135deg, rgba(34,211,238,0.15), rgba(96,165,250,0.16));
    border: 1px solid rgba(125,211,252,0.15); border-radius: 24px; padding: 22px 24px;
    margin-bottom: 16px; box-shadow: 0 14px 32px rgba(0,0,0,0.25);
}
.card {
    background: rgba(7,17,31,0.82); border: 1px solid rgba(125,211,252,0.15);
    border-radius: 20px; padding: 16px 18px; box-shadow: 0 10px 24px rgba(0,0,0,0.18);
}
.metric-label { color: #cbd5e1; font-size: 0.9rem; margin-bottom: 8px; }
.metric-value { color: #f8fafc; font-size: 2rem; font-weight: 800; line-height: 1.1; }
.metric-sub { color: #9fb3d1; font-size: 0.9rem; margin-top: 6px; }
.small-note { color: #dbeafe; font-size: 0.92rem; }

/* Force buttons to stay readable on mobile/light-mode oddities */
.stButton > button,
.stDownloadButton > button,
button[kind="primary"],
button[kind="secondary"],
div[data-testid="stFormSubmitButton"] > button,
form button {
    border-radius: 14px !important;
    border: 1px solid rgba(125,211,252,0.22) !important;
    font-weight: 800 !important;
    color: #ffffff !important;
    background: linear-gradient(135deg, #0ea5e9, #2563eb) !important;
    box-shadow: 0 8px 22px rgba(37,99,235,0.25) !important;
    opacity: 1 !important;
}
.stButton > button:hover,
.stDownloadButton > button:hover,
div[data-testid="stFormSubmitButton"] > button:hover,
form button:hover {
    filter: brightness(1.08) !important;
    color: #ffffff !important;
}
.stButton > button:focus,
.stDownloadButton > button:focus,
div[data-testid="stFormSubmitButton"] > button:focus,
form button:focus {
    outline: 2px solid #7dd3fc !important;
    outline-offset: 2px !important;
    color: #ffffff !important;
}
.stButton > button:disabled,
.stDownloadButton > button:disabled,
div[data-testid="stFormSubmitButton"] > button:disabled,
form button:disabled {
    background: linear-gradient(135deg, #1e3a8a, #1d4ed8) !important;
    color: #e2e8f0 !important;
    opacity: 0.95 !important;
}

.stTextInput input, .stSelectbox [data-baseweb="select"] > div {
    background: rgba(15,23,42,0.94) !important; color: #f8fafc !important;
    border: 1px solid rgba(148,163,184,0.25) !important; border-radius: 12px !important;
}
.stSelectbox [data-baseweb="select"] * { color: #f8fafc !important; }

/* Mobile-friendly radios instead of dropdown style issues */
div[role="radiogroup"] label {
    background: rgba(15,23,42,0.92) !important;
    border: 1px solid rgba(148,163,184,0.20) !important;
    border-radius: 12px !important;
    padding: 8px 14px !important;
    margin-right: 8px !important;
}
div[role="radiogroup"] * {
    color: #f8fafc !important;
}

.stSlider label, .stSlider span, .stSlider div { color: #e2e8f0 !important; }
[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #081124 0%, #0b1530 100%);
    border-right: 1px solid rgba(96,165,250,0.14);
}
[data-testid="stSidebar"] * { color: #f8fafc !important; }
</style>
""", unsafe_allow_html=True)

def get_conn():
    conn = sqlite3.connect(DB_PATH, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    return conn

def init_db():
    conn = get_conn()
    cur = conn.cursor()
    cur.execute("CREATE TABLE IF NOT EXISTS settings (key TEXT PRIMARY KEY, value TEXT NOT NULL)")
    cur.execute("""
        CREATE TABLE IF NOT EXISTS votes (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            voter_name TEXT,
            study_hours REAL NOT NULL,
            sleep_hours REAL NOT NULL,
            homework_pct REAL NOT NULL,
            attendance_pct REAL NOT NULL,
            avocado_flag INTEGER NOT NULL,
            created_at TEXT NOT NULL
        )
    """)
    defaults = {
        "accepting_votes": "1",
        "prediction_locked": "0",
        "last_prediction": "",
        "public_url": DEFAULT_PUBLIC_URL,
    }
    for k, v in defaults.items():
        cur.execute("INSERT OR IGNORE INTO settings (key, value) VALUES (?, ?)", (k, v))
    conn.commit()
    conn.close()

def get_setting(key, default=""):
    conn = get_conn()
    cur = conn.cursor()
    cur.execute("SELECT value FROM settings WHERE key=?", (key,))
    row = cur.fetchone()
    conn.close()
    return row["value"] if row else default

def set_setting(key, value):
    conn = get_conn()
    cur = conn.cursor()
    cur.execute(
        "INSERT INTO settings (key, value) VALUES (?, ?) ON CONFLICT(key) DO UPDATE SET value=excluded.value",
        (key, str(value))
    )
    conn.commit()
    conn.close()

def add_vote(voter_name, study_hours, sleep_hours, homework_pct, attendance_pct, avocado_flag):
    conn = get_conn()
    cur = conn.cursor()
    cur.execute("""
        INSERT INTO votes (voter_name, study_hours, sleep_hours, homework_pct, attendance_pct, avocado_flag, created_at)
        VALUES (?, ?, ?, ?, ?, ?, ?)
    """, (
        voter_name, float(study_hours), float(sleep_hours), float(homework_pct),
        float(attendance_pct), int(avocado_flag), datetime.utcnow().isoformat()
    ))
    conn.commit()
    conn.close()

def reset_votes():
    conn = get_conn()
    cur = conn.cursor()
    cur.execute("DELETE FROM votes")
    conn.commit()
    conn.close()
    set_setting("prediction_locked", "0")
    set_setting("last_prediction", "")
    set_setting("accepting_votes", "1")

def load_votes():
    conn = get_conn()
    df = pd.read_sql_query("SELECT * FROM votes ORDER BY id DESC", conn)
    conn.close()
    return df

@st.cache_resource
def train_model():
    rng = np.random.default_rng(SEED)
    n = 700
    study = rng.uniform(0, 25, size=n)
    sleep = rng.uniform(4, 10, size=n)
    homework = rng.uniform(20, 100, size=n)
    attendance = rng.uniform(60, 100, size=n)
    avocado = rng.integers(0, 2, size=n)
    math_grade = (
        35 + 1.6 * study + 2.4 * sleep + 0.22 * homework + 0.20 * attendance + 1.5 * avocado
        - 0.22 * np.maximum(0, sleep - 8.5) ** 2 + rng.normal(0, 4.5, size=n)
    )
    math_grade = np.clip(math_grade, 35, 100)
    X = pd.DataFrame({
        "study_hours": study,
        "sleep_hours": sleep,
        "homework_pct": homework,
        "attendance_pct": attendance,
        "avocado_flag": avocado
    })
    model = RandomForestRegressor(n_estimators=250, max_depth=8, random_state=SEED)
    model.fit(X, math_grade)
    return model

def numeric_to_letter(score: float) -> str:
    if score >= 86:
        return "A"
    elif score >= 73:
        return "B"
    elif score >= 60:
        return "C"
    elif score >= 50:
        return "D"
    else:
        return "E"

def predict_from_consensus(votes_df):
    if votes_df.empty:
        return None, None, None, None
    avg = {
        "study_hours": votes_df["study_hours"].mean(),
        "sleep_hours": votes_df["sleep_hours"].mean(),
        "homework_pct": votes_df["homework_pct"].mean(),
        "attendance_pct": votes_df["attendance_pct"].mean(),
        "avocado_flag": round(votes_df["avocado_flag"].mean()),
    }
    model = train_model()
    X_new = pd.DataFrame([avg])
    pred = float(model.predict(X_new)[0])
    letter = numeric_to_letter(pred)
    imp = pd.DataFrame({"feature": X_new.columns, "importance": model.feature_importances_}).sort_values("importance", ascending=False)
    return pred, letter, avg, imp

def make_qr_bytes(url: str):
    qr = qrcode.QRCode(box_size=9, border=1)
    qr.add_data(url)
    qr.make(fit=True)
    img = qr.make_image(fill_color="black", back_color="white")
    bio = io.BytesIO()
    img.save(bio, format="PNG")
    return bio.getvalue()

def grade_gauge(pred, letter):
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=pred,
        number={"suffix": "%", "font": {"size": 34}},
        title={"text": f"<span style='font-size:28px'><b>Letter grade: {letter}</b></span>"},
        gauge={
            "axis": {"range": [0, 100]},
            "bar": {"color": "#38bdf8"},
            "steps": [
                {"range": [0, 50], "color": "rgba(239,68,68,0.25)"},
                {"range": [50, 60], "color": "rgba(249,115,22,0.25)"},
                {"range": [60, 73], "color": "rgba(245,158,11,0.25)"},
                {"range": [73, 86], "color": "rgba(34,197,94,0.25)"},
                {"range": [86, 100], "color": "rgba(16,185,129,0.35)"},
            ],
        }
    ))
    fig.update_layout(template="plotly_dark", height=320, margin=dict(l=20, r=20, t=60, b=20),
                      paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                      font=dict(color="#f8fafc"), title="Predicted math result")
    return fig

def consensus_bar(avg_dict):
    labels = ["Study hrs / week", "Sleep hrs / night", "Homework %", "Attendance %", "Avocado vote"]
    values = [avg_dict["study_hours"], avg_dict["sleep_hours"], avg_dict["homework_pct"], avg_dict["attendance_pct"], avg_dict["avocado_flag"] * 100]
    max_scale = [25, 10, 100, 100, 100]
    normalized = [v / m * 100 for v, m in zip(values, max_scale)]
    fig = go.Figure(go.Bar(x=normalized, y=labels, orientation="h", marker=dict(color="#22c55e")))
    fig.update_layout(template="plotly_dark", height=320, margin=dict(l=20, r=20, t=30, b=20),
                      paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                      font=dict(color="#f8fafc"), title="Live class consensus")
    fig.update_xaxes(title="Relative scale (%)", gridcolor="rgba(148,163,184,0.16)")
    return fig

def feature_importance_chart(imp_df):
    labels = {
        "study_hours": "Study hrs/wk",
        "sleep_hours": "Sleep hrs/night",
        "homework_pct": "Homework %",
        "attendance_pct": "Attendance %",
        "avocado_flag": "Avocado"
    }
    d = imp_df.copy()
    d["feature"] = d["feature"].map(labels).fillna(d["feature"])
    fig = go.Figure(go.Bar(x=d["importance"], y=d["feature"], orientation="h", marker=dict(color="#f59e0b")))
    fig.update_layout(template="plotly_dark", height=320, margin=dict(l=20, r=20, t=30, b=20),
                      paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                      font=dict(color="#f8fafc"), title="What the model cares about")
    fig.update_xaxes(title="Importance", gridcolor="rgba(148,163,184,0.16)")
    return fig

def render_vote_page():
    st_autorefresh(interval=4000, limit=None, key="vote_refresh")
    accepting_votes = get_setting("accepting_votes", "1") == "1"
    st.markdown("""
    <div class="hero">
        <div style="font-size:2.1rem;font-weight:900;">📱 Audience Input Panel</div>
        <div class="small-note" style="margin-top:8px;">Vote on the student profile. Your choices feed into the live classroom dashboard.</div>
    </div>
    """, unsafe_allow_html=True)
    if not accepting_votes:
        st.warning("Voting is closed. Watch the main screen for the final prediction.")
        return
    with st.form("vote_form", clear_on_submit=True):
        voter_name = st.text_input("Your name or nickname", max_chars=30, placeholder="Optional")
        study_hours = st.slider("Hours studied per week", 0, 25, 10)
        sleep_hours = st.slider("Sleep hours per night", 4, 10, 7)
        homework_pct = st.slider("Homework completion (%)", 0, 100, 80)
        attendance_pct = st.slider("Attendance (%)", 50, 100, 90)
        avocado = st.radio("Eats avocados?", ["No", "Yes"], horizontal=True)
        submitted = st.form_submit_button("Submit my vote", use_container_width=True)
    if submitted:
        add_vote(voter_name.strip(), study_hours, sleep_hours, homework_pct, attendance_pct, 1 if avocado == "Yes" else 0)
        st.success("Vote submitted. Look at the main dashboard to see it update.")

def render_dashboard():
    st_autorefresh(interval=3000, limit=None, key="dashboard_refresh")
    votes_df = load_votes()
    st.markdown("""
    <div class="hero">
        <div style="font-size:2.25rem;font-weight:900;">📚 Live Math Grade Predictor</div>
        <div class="small-note" style="margin-top:8px;">Let the audience build the student profile in real time. Then freeze the inputs and reveal the ML prediction.</div>
    </div>
    """, unsafe_allow_html=True)

    with st.sidebar:
        st.markdown("### Presenter controls")
        accepting_votes = get_setting("accepting_votes", "1") == "1"
        public_url = st.text_input("Public app URL", value=get_setting("public_url", DEFAULT_PUBLIC_URL))
        if public_url != get_setting("public_url", DEFAULT_PUBLIC_URL):
            set_setting("public_url", public_url)
        st.markdown(f"Voting status: **{'OPEN' if accepting_votes else 'LOCKED'}**")
        c1, c2 = st.columns(2)
        with c1:
            if st.button("Open voting", use_container_width=True):
                set_setting("accepting_votes", "1")
                set_setting("prediction_locked", "0")
                st.rerun()
        with c2:
            if st.button("Freeze voting", use_container_width=True):
                set_setting("accepting_votes", "0")
                st.rerun()
        if st.button("Run prediction", use_container_width=True):
            pred, letter, avg, _ = predict_from_consensus(votes_df)
            if pred is not None:
                payload = {"pred": round(pred, 1), "letter": letter, "avg": avg, "created_at": datetime.utcnow().isoformat()}
                set_setting("last_prediction", json.dumps(payload))
                set_setting("prediction_locked", "1")
                set_setting("accepting_votes", "0")
                st.rerun()
        if st.button("Reset class session", use_container_width=True):
            reset_votes()
            st.rerun()
        st.caption("Project the dashboard and let students scan the QR code to vote.")

    public_url = get_setting("public_url", DEFAULT_PUBLIC_URL).strip().rstrip("/")
    vote_url = public_url + ("&" if "?" in public_url else "?") + urlencode({"mode": "vote"})
    qr_bytes = make_qr_bytes(vote_url)

    last_prediction_raw = get_setting("last_prediction", "")
    pred_payload = json.loads(last_prediction_raw) if last_prediction_raw else None
    total_votes = len(votes_df)
    accepting_votes = get_setting("accepting_votes", "1") == "1"

    m1, m2, m3, m4 = st.columns(4)
    cards = [
        ("Votes received", str(total_votes), "Audience submissions so far"),
        ("Voting", "Open" if accepting_votes else "Frozen", "Use freeze before revealing prediction"),
        ("Predicted grade", pred_payload["letter"] if pred_payload else "—", "Letter grade appears after prediction"),
        ("Avocado votes", str(int(votes_df["avocado_flag"].sum()) if not votes_df.empty else 0), "Because every good demo needs a weird feature"),
    ]
    for col, (a,b,c) in zip([m1,m2,m3,m4], cards):
        with col:
            st.markdown(f'<div class="card"><div class="metric-label">{a}</div><div class="metric-value">{b}</div><div class="metric-sub">{c}</div></div>', unsafe_allow_html=True)

    left, right = st.columns([2.1, 1.0], gap="large")
    with left:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        if votes_df.empty:
            st.info("No votes yet. Ask students to scan the QR code and submit their choices.")
        else:
            avg_live = {
                "study_hours": votes_df["study_hours"].mean(),
                "sleep_hours": votes_df["sleep_hours"].mean(),
                "homework_pct": votes_df["homework_pct"].mean(),
                "attendance_pct": votes_df["attendance_pct"].mean(),
                "avocado_flag": round(votes_df["avocado_flag"].mean()),
            }
            c1, c2 = st.columns(2)
            with c1:
                st.plotly_chart(consensus_bar(avg_live), use_container_width=True)
            with c2:
                if pred_payload:
                    _, _, _, imp_df = predict_from_consensus(votes_df)
                    st.plotly_chart(feature_importance_chart(imp_df), use_container_width=True)
                else:
                    st.markdown("### Prediction pending")
                    st.markdown("Freeze voting and click **Run prediction** to reveal the forecast.")
                    st.dataframe(
                        votes_df[["voter_name","study_hours","sleep_hours","homework_pct","attendance_pct","avocado_flag"]]
                        .rename(columns={"avocado_flag": "avocado_yes"}).head(12),
                        use_container_width=True, hide_index=True
                    )
        st.markdown('</div>', unsafe_allow_html=True)

        if pred_payload:
            st.markdown('<div class="card" style="margin-top:16px;">', unsafe_allow_html=True)
            p1, p2 = st.columns([1.1, 1.2])
            with p1:
                st.plotly_chart(grade_gauge(pred_payload["pred"], pred_payload["letter"]), use_container_width=True)
            with p2:
                avg = pred_payload["avg"]
                summary_df = pd.DataFrame({
                    "Feature": ["Study hrs/wk","Sleep hrs/night","Homework %","Attendance %","Avocado"],
                    "Consensus value": [
                        round(avg["study_hours"], 1),
                        round(avg["sleep_hours"], 1),
                        round(avg["homework_pct"], 1),
                        round(avg["attendance_pct"], 1),
                        "Yes" if avg["avocado_flag"] >= 0.5 else "No",
                    ]
                })
                st.markdown(f"### Frozen class consensus → **{pred_payload['letter']}**")
                st.dataframe(summary_df, use_container_width=True, hide_index=True)
                st.caption(f"Underlying predicted percentage: {pred_payload['pred']:.1f}%")
            st.markdown('</div>', unsafe_allow_html=True)

    with right:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown("### Scan to vote")
        st.image(qr_bytes, use_container_width=True)
        st.caption(vote_url)
        st.download_button("Download QR PNG", data=qr_bytes, file_name="math_grade_vote_qr.png", mime="image/png")
        st.markdown("---")
        st.markdown("### How this works")
        st.markdown("- Students scan the code\n- They submit a feature profile\n- The dashboard updates live\n- You freeze submissions\n- The model predicts a letter grade")
        st.markdown('</div>', unsafe_allow_html=True)

init_db()
mode = st.query_params.get("mode", "dashboard")
if mode == "vote":
    render_vote_page()
else:
    render_dashboard()
