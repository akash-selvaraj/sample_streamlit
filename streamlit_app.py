# streamlit_app.py (fixed)
# --- Legal Survey Analysis App: SaaS Upgrade ---
# Changes in this version:
# - Replaced st.experimental_get_query_params -> st.query_params
# - Removed st.experimental_set_query_params; now using st.query_params.clear()
# - Hardened OAuth state handling with Fernet-encrypted state token to avoid mismatches
# - Minor robustness tweaks (comments show UPDATED sections)

from __future__ import annotations

import os
import io
import json
import textwrap
import itertools
from datetime import datetime, timedelta
from typing import List, Tuple, Dict, Any, Optional

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import streamlit as st
import requests

# Auth / Crypto / DB
from authlib.integrations.requests_client import OAuth2Session
from cryptography.fernet import Fernet
from pymongo import MongoClient, ReturnDocument

# Stats
from scipy import stats

# -------------- CONFIG --------------
st.set_page_config(page_title="Legal Survey Analysis App", layout="wide")

APP_NAME = "Legal Survey Analysis App"
SUBSCRIPTION_PRICE_INR = 119

# Secrets expected in Streamlit Cloud (st.secrets)
# st.secrets["MONGODB_URI"]
# st.secrets["FERNET_KEY"]  # generate once: Fernet.generate_key().decode()
# st.secrets["GOOGLE_CLIENT_ID"], st.secrets["GOOGLE_CLIENT_SECRET"], st.secrets["OAUTH_REDIRECT_URI"]
# st.secrets["CASHFREE_APP_ID"], st.secrets["CASHFREE_SECRET_KEY"], st.secrets.get("CASHFREE_ENV", "SANDBOX")

# -------------- UTILITIES --------------

def get_fernet() -> Fernet:
    key = st.secrets.get("FERNET_KEY")
    if not key:
        st.error("FERNET_KEY missing in secrets.")
        st.stop()
    return Fernet(key.encode() if isinstance(key, str) else key)


def encrypt_str(s: str) -> str:
    f = get_fernet()
    return f.encrypt(s.encode()).decode()


def decrypt_str(s: str) -> str:
    f = get_fernet()
    return f.decrypt(s.encode()).decode()


def get_db():
    uri = st.secrets.get("MONGODB_URI")
    if not uri:
        st.error("MongoDB URI missing in secrets.")
        st.stop()
    client = MongoClient(uri)
    return client["legal_survey_app"]


def get_users_col():
    return get_db()["users"]


# -------------- AUTH (GOOGLE OAUTH) --------------

def get_oauth_session() -> OAuth2Session:
    client_id = st.secrets.get("GOOGLE_CLIENT_ID")
    client_secret = st.secrets.get("GOOGLE_CLIENT_SECRET")
    redirect_uri = st.secrets.get("OAUTH_REDIRECT_URI")
    if not all([client_id, client_secret, redirect_uri]):
        st.error("Google OAuth secrets missing.")
        st.stop()
    return OAuth2Session(
        client_id,
        client_secret,
        scope="openid email profile",
        redirect_uri=redirect_uri,
    )


def login_ui():
    st.markdown("### Login")
    oauth = get_oauth_session()
    authorization_endpoint = "https://accounts.google.com/o/oauth2/v2/auth"

    # UPDATED: generate raw state, store in session, send ENCRYPTED state in URL
    f = get_fernet()
    raw_state = st.session_state.get("oauth_state_raw")
    if not raw_state:
        import secrets as _secrets
        raw_state = _secrets.token_urlsafe(16)
        st.session_state["oauth_state_raw"] = raw_state
    enc_state = f.encrypt(raw_state.encode()).decode()
    st.session_state["oauth_state_enc"] = enc_state

    # Authlib lets us pass extra params to authorization URL
    uri, _generated_state = oauth.create_authorization_url(
        authorization_endpoint,
        prompt="consent",
        state=enc_state,  # we control state token
    )
    st.link_button("Sign in with Google", uri, use_container_width=True)


def handle_oauth_redirect():
    params = st.query_params
    code = params.get("code")
    state = params.get("state")
    if not code or not state:
        return False

    # ✅ Compare directly to stored encrypted state
    if state != st.session_state.get("oauth_state_enc"):
        st.warning("State mismatch in OAuth flow. Try logging in again.")
        return False

    oauth = get_oauth_session()
    token_endpoint = "https://oauth2.googleapis.com/token"
    token = oauth.fetch_token(
        token_endpoint,
        code=code,
        grant_type="authorization_code",
    )

    # Fetch userinfo
    resp = oauth.get("https://openidconnect.googleapis.com/v1/userinfo", token=token)
    if resp.status_code != 200:
        st.error("Failed to fetch user info.")
        return False
    profile = resp.json()

    # Persist minimal profile in session
    st.session_state["user"] = {
        "email": profile.get("email"),
        "name": profile.get("name"),
        "picture": profile.get("picture"),
        "sub": profile.get("sub"),
        "token": token,
    }

    # Upsert user in MongoDB
    users = get_users_col()
    users.find_one_and_update(
        {"google_sub": profile.get("sub")},
        {
            "$setOnInsert": {
                "created_at": datetime.utcnow(),
                "email": profile.get("email"),
                "name": profile.get("name"),
                "google_sub": profile.get("sub"),
                "subscription_active": False,
            },
            "$set": {"last_login": datetime.utcnow()},
        },
        upsert=True,
        return_document=ReturnDocument.AFTER,
    )

    # ✅ Clear query string safely
    try:
        st.query_params.clear()
    except Exception:
        pass

    return True


def current_user() -> Optional[Dict[str, Any]]:
    return st.session_state.get("user")


# -------------- CASHFREE SUBSCRIPTIONS --------------

CASHFREE_ENV = st.secrets.get("CASHFREE_ENV", "SANDBOX").upper()
CF_BASE = "https://sandbox.cashfree.com/pg" if CASHFREE_ENV == "SANDBOX" else "https://api.cashfree.com/pg"


def cf_headers():
    return {
        "x-client-id": st.secrets.get("CASHFREE_APP_ID", ""),
        "x-client-secret": st.secrets.get("CASHFREE_SECRET_KEY", ""),
        "x-api-version": "2022-09-01",
        "Content-Type": "application/json",
    }


def create_cashfree_order(amount_inr: int, customer_id: str, customer_email: str, customer_phone: str) -> Optional[dict]:
    payload = {
        "order_id": f"order_{customer_id}_{int(datetime.utcnow().timestamp())}",
        "order_amount": float(amount_inr),
        "order_currency": "INR",
        "customer_details": {
            "customer_id": customer_id,
            "customer_email": customer_email,
            "customer_phone": customer_phone or "9999999999",
        },
        "order_note": f"{APP_NAME} Monthly Subscription",
        "order_meta": {
            "return_url": st.secrets.get("OAUTH_REDIRECT_URI")
        },
    }
    try:
        r = requests.post(f"{CF_BASE}/orders", headers=cf_headers(), data=json.dumps(payload), timeout=20)
        if r.status_code in (200, 201):
            return r.json()
        else:
            st.error(f"Cashfree order error: {r.status_code} {r.text}")
            return None
    except Exception as e:
        st.error(f"Cashfree order exception: {e}")
        return None


def get_cashfree_order(order_id: str) -> Optional[dict]:
    try:
        r = requests.get(f"{CF_BASE}/orders/{order_id}", headers=cf_headers(), timeout=20)
        if r.status_code == 200:
            return r.json()
        st.error(f"Cashfree status error: {r.status_code} {r.text}")
        return None
    except Exception as e:
        st.error(f"Cashfree status exception: {e}")
        return None


# -------------- DATA HELPERS --------------

@st.cache_data(show_spinner=False)
def load_tabular_file(uploaded_file) -> pd.DataFrame:
    name = uploaded_file.name
    ext = name.lower().split(".")[-1]

    content = uploaded_file.read()
    bio = io.BytesIO(content)

    if ext in ["xlsx", "xls"]:
        return pd.read_excel(bio)
    elif ext == "csv":
        return pd.read_csv(bio, encoding="utf-8")
    elif ext == "tsv":
        return pd.read_csv(bio, sep="\t", encoding="utf-8")
    elif ext == "odt":
        try:
            from odf.opendocument import load as opendocument_load
            from odf.table import Table, TableRow, TableCell
            from odf.text import P
        except Exception:
            st.warning("odfpy not installed. ODT files are not supported in this deployment.")
            raise
        doc = opendocument_load(bio)
        data_rows = []
        for elem in doc.getElementsByType(Table):
            for row in elem.getElementsByType(TableRow):
                row_data = []
                for cell in row.getElementsByType(TableCell):
                    cell_text = ''.join(p.firstChild.data if p.firstChild else '' for p in cell.getElementsByType(P))
                    row_data.append(cell_text)
                data_rows.append(row_data)
        return pd.DataFrame(data_rows[1:], columns=data_rows[0] if data_rows else [])
    else:
        raise ValueError(f"Unsupported file format: {ext}")


def wrap_labels(ax, width=18):
    labels = []
    for label in ax.get_xticklabels():
        text = label.get_text()
        wrapped = "\n".join(textwrap.wrap(text, width=width)) if text else ""
        labels.append(wrapped)
    ax.set_xticklabels(labels, rotation=0, ha='center')


def auto_fig_width(n_cats: int, base=6.0, per_cat=0.4, maxw=20.0) -> float:
    return float(min(max(base, base + n_cats * per_cat), maxw))


def compute_crosstab(df: pd.DataFrame, row: str, col: str, values: Optional[str] = None, normalize: Optional[str] = None) -> pd.DataFrame:
    if values and values in df.columns:
        ct = pd.crosstab(df[row], [df[col], df[values]], margins=True, dropna=False, normalize=normalize)
    else:
        ct = pd.crosstab(df[row], df[col], margins=True, dropna=False, normalize=normalize)
    return ct


def df_to_excel_download(df_dict: Dict[str, pd.DataFrame], filename: str = "tables.xlsx"):
    bio = io.BytesIO()
    with pd.ExcelWriter(bio, engine="xlsxwriter") as writer:
        for sheet, df in df_dict.items():
            df.to_excel(writer, index=True, sheet_name=sheet[:31])
    bio.seek(0)
    st.download_button("Download tables as Excel", data=bio, file_name=filename, mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")


# -------------- RAG over Data --------------

def build_rag_chunks(df: pd.DataFrame, max_rows: int = 200) -> List[str]:
    sdf = df.sample(min(len(df), max_rows), random_state=42) if len(df) > max_rows else df.copy()
    chunks = []
    for col in sdf.columns:
        series = sdf[col]
        if pd.api.types.is_numeric_dtype(series):
            desc = series.describe().to_dict()
            chunk = f"Column: {col} (numeric). Summary: {json.dumps(desc)}"
        else:
            vc = series.astype(str).value_counts().head(20).to_dict()
            chunk = f"Column: {col} (categorical). Top values: {json.dumps(vc)}"
        chunks.append(chunk)
    for _, row in sdf.head(30).iterrows():
        chunks.append(f"Row: {json.dumps(row.to_dict(), default=str)}")
    return chunks


@st.cache_resource(show_spinner=False)
def get_vectorizer():
    from sklearn.feature_extraction.text import TfidfVectorizer
    return TfidfVectorizer(max_features=4096)


def rag_retrieve(query: str, chunks: List[str], top_k: int = 10) -> List[str]:
    if not chunks:
        return []
    vec = get_vectorizer()
    X = vec.fit_transform(chunks)
    q = vec.transform([query])
    sims = (X @ q.T).toarray().ravel()
    idx = np.argsort(-sims)[:top_k]
    return [chunks[i] for i in idx]


# -------------- HYPOTHESIS TESTING --------------

TESTS_HELP = {
    "Auto": "Automatically selects Chi-square for two categoricals; t-test for numeric vs 2 groups; ANOVA for numeric vs >2 groups; F-test for two variances.",
    "Chi-square": "Association test for two categorical variables.",
    "t-test (independent)": "Compare means of a numeric variable across two independent groups.",
    "ANOVA": "Compare means of a numeric variable across more than two groups.",
    "F-test (variances)": "Compare variances of two numeric samples.",
}


def auto_select_test(df: pd.DataFrame, y: str, x: Optional[str]) -> str:
    if x is None:
        return "t-test (independent)"
    y_is_num = pd.api.types.is_numeric_dtype(df[y])
    x_is_num = pd.api.types.is_numeric_dtype(df[x])
    if (not y_is_num) and (not x_is_num):
        return "Chi-square"
    if y_is_num and (not x_is_num):
        k = df[x].nunique(dropna=True)
        return "t-test (independent)" if k == 2 else "ANOVA"
    if (not y_is_num) and x_is_num:
        k = df[y].nunique(dropna=True)
        return "t-test (independent)" if k == 2 else "ANOVA"
    return "F-test (variances)"


def run_test(df: pd.DataFrame, test_name: str, y: str, x: Optional[str]) -> Dict[str, Any]:
    res = {"test": test_name, "p_value": None, "statistic": None, "notes": ""}

    if test_name == "Chi-square":
        tbl = pd.crosstab(df[y], df[x])
        chi2, p, dof, expected = stats.chi2_contingency(tbl)
        res.update({"statistic": chi2, "p_value": p, "dof": dof, "table": tbl})
        return res

    if test_name == "t-test (independent)":
        if x is None:
            res["notes"] = "x (group) is required for t-test."
            return res
        groups = [g.dropna().values for _, g in df.groupby(x)[y]]
        groups = [g for g in groups if len(g) > 0]
        if len(groups) != 2:
            res["notes"] = "t-test requires exactly 2 groups."
            return res
        t, p = stats.ttest_ind(groups[0], groups[1], equal_var=False, nan_policy='omit')
        res.update({"statistic": t, "p_value": p, "group_sizes": [len(g) for g in groups]})
        return res

    if test_name == "ANOVA":
        if x is None:
            res["notes"] = "x (group) is required for ANOVA."
            return res
        groups = [g.dropna().values for _, g in df.groupby(x)[y]]
        if len(groups) < 3:
            res["notes"] = "ANOVA requires 3+ groups."
            return res
        f, p = stats.f_oneway(*groups)
        res.update({"statistic": f, "p_value": p, "group_sizes": [len(g) for g in groups]})
        return res

    if test_name == "F-test (variances)":
        if x is None:
            res["notes"] = "Select two numeric variables (y and x) for F-test."
            return res
        a = df[y].dropna().values
        b = df[x].dropna().values
        if len(a) < 2 or len(b) < 2:
            res["notes"] = "Need at least 2 observations per group for F-test."
            return res
        f = np.var(a, ddof=1) / np.var(b, ddof=1)
        dfn = len(a) - 1
        dfd = len(b) - 1
        p = 2 * min(stats.f.cdf(f, dfn, dfd), 1 - stats.f.cdf(f, dfn, dfd))
        res.update({"statistic": f, "p_value": p, "df": (dfn, dfd)})
        return res

    res["notes"] = "Unknown test."
    return res


# -------------- GEMINI (PER-USER KEY) --------------

GEMINI_MODEL = "gemini-1.5-flash-latest"
GEMINI_URL = f"https://generativelanguage.googleapis.com/v1beta/models/{GEMINI_MODEL}:generateContent"


def get_user_record() -> Optional[dict]:
    user = current_user()
    if not user:
        return None
    users = get_users_col()
    return users.find_one({"google_sub": user["sub"]})


def save_user_gemini_key(api_key_plain: str):
    user = current_user()
    if not user:
        return
    users = get_users_col()
    enc = encrypt_str(api_key_plain)
    users.update_one({"google_sub": user["sub"]}, {"$set": {"gemini_key": enc}})


def get_user_gemini_key() -> Optional[str]:
    rec = get_user_record()
    if not rec or not rec.get("gemini_key"):
        return None
    try:
        return decrypt_str(rec["gemini_key"])
    except Exception:
        return None


def gemini_generate_json(prompt: str, api_key: str, max_output_tokens: int = 2048, temperature: float = 0.4) -> dict:
    headers = {"Content-Type": "application/json"}
    payload = {
        "contents": [{"parts": [{"text": prompt}]}],
        "generationConfig": {
            "response_mime_type": "application/json",
            "temperature": temperature,
            "maxOutputTokens": max_output_tokens,
        },
    }
    r = requests.post(f"{GEMINI_URL}?key={api_key}", headers=headers, data=json.dumps(payload), timeout=60)
    r.raise_for_status()
    result = r.json()
    try:
        text = result['candidates'][0]['content']['parts'][0]['text']
        return json.loads(text)
    except Exception:
        return {"answer": "(Model returned an unexpected format)", "code": None}


# -------------- APP BODY --------------

st.title(APP_NAME)

# OAuth callback handling first, then session check
if handle_oauth_redirect():
    st.toast("Logged in via Google ✅", icon="✅")

user = current_user()

if not user:
    with st.container():
        st.info("Please sign in to continue.")
        login_ui()
        st.stop()

# Fetch DB record and subscription state
user_rec = get_user_record() or {}
subscription_active = bool(user_rec.get("subscription_active", False))

# Sidebar: Profile & Subscription
with st.sidebar:
    if user.get("picture"):
        st.image(user.get("picture"), width=64)
    st.markdown(f"**{user.get('name','User')}**\n\n{user.get('email','')}")
    st.markdown("---")
    if subscription_active:
        st.success("Subscription: Active ✅")
    else:
        st.warning("Subscription: Inactive")
        phone = st.text_input("Phone for Cashfree", value="9999999999")
        if st.button(f"Subscribe ₹{SUBSCRIPTION_PRICE_INR}/month", use_container_width=True):
            order = create_cashfree_order(SUBSCRIPTION_PRICE_INR, user["sub"], user.get("email",""), phone)
            if order and order.get("payment_session_id"):
                st.session_state["cf_order_id"] = order.get("order_id")
                st.session_state["cf_payment_session_id"] = order.get("payment_session_id")
                if order.get("payment_link"):
                    st.markdown(f"[Proceed to Cashfree Checkout]({order.get('payment_link')})")
                else:
                    st.info("Order created. Complete payment in the opened Cashfree page.")
            elif order and order.get("order_id"):
                st.markdown(f"Order created. ID: `{order['order_id']}`")

    if st.button("Verify Subscription Status"):
        oid = st.session_state.get("cf_order_id")
        if not oid:
            st.info("Create an order first or complete payment.")
        else:
            data = get_cashfree_order(oid)
            if data and data.get("order_status") == "PAID":
                st.success("Payment verified! Subscription activated.")
                get_users_col().update_one({"google_sub": user["sub"]}, {"$set": {"subscription_active": True, "subscription_updated": datetime.utcnow()}})
            else:
                st.info("Payment not captured yet. If you just paid, try again shortly.")

    st.markdown("---")
    # Gemini Key (encrypted per user)
    current_key = "✔️ Stored" if get_user_gemini_key() else "—"
    st.caption(f"Gemini API Key: {current_key}")
    if st.button("Update Gemini API Key"):
        with st.modal("Set Gemini API Key"):
            k = st.text_input("Enter your Gemini API Key", type="password")
            if st.button("Save", type="primary") and k:
                save_user_gemini_key(k)
                st.success("Saved (encrypted). Close this dialog.")

# -------------- FILE UPLOAD --------------
with st.container(border=True):
    st.subheader("Upload Data")
    uploaded = st.file_uploader("Upload your tabular data file", type=["xlsx","xls","csv","tsv","odt"])
    if uploaded is not None:
        try:
            df = load_tabular_file(uploaded)
            st.session_state["data"] = df
            st.success(f"Loaded shape: {df.shape}")
        except Exception as e:
            st.error(f"Error processing file: {e}")

if "data" not in st.session_state:
    st.info("Upload a dataset to get started.")
    st.stop()

df: pd.DataFrame = st.session_state["data"]
num_cols = df.select_dtypes(include=np.number).columns.tolist()
cat_cols = df.select_dtypes(exclude=[np.number]).columns.tolist()
all_cols = list(df.columns)

# Precompute RAG chunks for AI Tab
st.session_state["rag_chunks"] = build_rag_chunks(df)

# -------------- TABS --------------

tab_preview, tab_viz, tab_summary, tab_hypo, tab_chat, tab_settings = st.tabs([
    "Data Preview", "Visualizations", "Summaries", "Hypothesis Testing", "AI Chat", "Settings"
])

# --- Data Preview ---
with tab_preview:
    st.subheader("Preview")
    st.dataframe(df.head(100), use_container_width=True)

    with st.expander("Quick Column Info"):
        info = pd.DataFrame({
            "column": df.columns,
            "dtype": [str(df[c].dtype) for c in df.columns],
            "n_unique": [df[c].nunique(dropna=True) for c in df.columns],
            "n_missing": [df[c].isna().sum() for c in df.columns],
        })
        st.dataframe(info, use_container_width=True)

# --- Visualizations ---
with tab_viz:
    st.subheader("Custom Visualization")
    col_left, col_right = st.columns([2,1])

    with col_left:
        analysis_type = st.selectbox("Analysis Type", ["Univariate", "Bivariate"], key="viz_type")
        y_mode = st.radio("Y-axis", ["Count", "Percentage"], horizontal=True)
        color_custom = st.checkbox("Customize bar color")
        bar_color = st.color_picker("Pick bar color", "#87CEEB") if color_custom else None

    with col_right:
        label_wrap = st.slider("X label wrap width", min_value=8, max_value=30, value=16)
        width_base = st.slider("Base figure width", 4.0, 12.0, 8.0, 0.5)
        width_per_cat = st.slider("Width per category", 0.1, 1.0, 0.4, 0.05)

    if analysis_type == "Univariate":
        selected_col = st.selectbox("Column", all_cols, key="uni_col")
        if st.button("Generate Graph", type="primary"):
            fig, ax = plt.subplots(figsize=(auto_fig_width(df[selected_col].nunique(), base=width_base, per_cat=width_per_cat), 5))

            if selected_col in cat_cols:
                vc = df[selected_col].astype(str).value_counts(dropna=False)
            else:
                binned = pd.cut(df[selected_col], bins=10)
                vc = binned.value_counts().sort_index()
                vc.index = vc.index.astype(str)

            total = vc.sum()
            y_vals = vc.values if y_mode == "Count" else (vc.values / total * 100.0)
            ylabel = "Count" if y_mode == "Count" else "Percentage (%)"

            bars = ax.bar(vc.index, y_vals, color=bar_color or ("#87CEEB" if selected_col in cat_cols else "#90EE90"))

            labels = [
                f"{int(v.get_height())}" if y_mode=="Count" else f"{v.get_height():.1f}%"
                for v in bars
            ]
            ax.bar_label(bars, labels=labels, label_type='edge', color='black', fontsize=9)

            ax.set_ylabel(ylabel)
            ax.set_xlabel(selected_col)
            plt.setp(ax.get_xticklabels(), rotation=0, ha="center")
            wrap_labels(ax, width=label_wrap)
            fig.tight_layout()
            st.pyplot(fig)

    else:
        col1 = st.selectbox("First Column", all_cols, key='bi_col1')
        col2 = st.selectbox("Second Column", all_cols, key='bi_col2')
        if st.button("Generate Graph", type="primary"):
            if col1 == col2:
                st.warning("Please select two different columns.")
            else:
                plot_df = df.copy()
                if col1 in num_cols:
                    plot_df[col1] = pd.cut(plot_df[col1], bins=5, labels=[f"Bin {i+1}" for i in range(5)])
                if col2 in num_cols:
                    plot_df[col2] = pd.cut(plot_df[col2], bins=5, labels=[f"Bin {i+1}" for i in range(5)])

                ct = pd.crosstab(plot_df[col1].astype(str), plot_df[col2].astype(str))
                totals = ct.values.sum()
                plot_ct = ct if y_mode == "Count" else (ct / totals * 100.0)

                fig, ax = plt.subplots(figsize=(auto_fig_width(len(ct.index)*len(ct.columns), base=width_base, per_cat=width_per_cat), 6))
                for i, col in enumerate(plot_ct.columns):
                    vals = plot_ct[col].values
                    xpos = np.arange(len(plot_ct.index)) + i*0.8/len(plot_ct.columns)
                    width = 0.8/len(plot_ct.columns)
                    ax.bar(xpos, vals, width=width, label=str(col), color=bar_color if color_custom else None)
                ax.set_xticks(np.arange(len(plot_ct.index)) + 0.4)
                ax.set_xticklabels(plot_ct.index.astype(str).tolist())
                wrap_labels(ax, width=label_wrap)
                ax.set_ylabel("Count" if y_mode=="Count" else "Percentage (%)")
                ax.set_xlabel(col1)
                ax.legend(title=col2, bbox_to_anchor=(1.02, 1), loc='upper left')
                fig.tight_layout()
                st.pyplot(fig)

# --- Summaries ---
with tab_summary:
    st.subheader("Summary Statistics (on demand)")
    target_cols = st.multiselect("Choose columns for summary", all_cols, default=num_cols[:5])
    if st.button("Compute Summary") and target_cols:
        rows = []
        for c in target_cols:
            s = df[c]
            if pd.api.types.is_numeric_dtype(s):
                rows.append({
                    "column": c,
                    "type": "numeric",
                    "mean": s.mean(),
                    "median": s.median(),
                    "mode": s.mode().iloc[0] if not s.mode().empty else np.nan,
                    "std": s.std(),
                    "min": s.min(),
                    "max": s.max(),
                    "missing": s.isna().sum(),
                })
            else:
                m = s.mode()
                rows.append({
                    "column": c,
                    "type": "categorical",
                    "mean": np.nan,
                    "median": np.nan,
                    "mode": m.iloc[0] if not m.empty else np.nan,
                    "std": np.nan,
                    "min": np.nan,
                    "max": np.nan,
                    "missing": s.isna().sum(),
                })
        out = pd.DataFrame(rows)
        st.dataframe(out, use_container_width=True)

    st.markdown("---")
    st.subheader("Two-way Split Tables with Totals")
    row_var = st.selectbox("Row variable", all_cols, key="row_var")
    col_var = st.selectbox("Column variable", all_cols, key="col_var")
    val_var = st.selectbox("Optional outcome variable (e.g., Bought)", ["(none)"] + all_cols, key="val_var")
    normalize = st.selectbox("Show as", ["Counts", "Percentages (overall)", "Percentages by row", "Percentages by column"]) 

    if st.button("Generate Table"):
        norm = None
        if normalize == "Percentages (overall)":
            norm = "all"
        elif normalize == "Percentages by row":
            norm = "index"
        elif normalize == "Percentages by column":
            norm = "columns"
        ct = compute_crosstab(df, row_var, col_var, values=None if val_var == "(none)" else val_var, normalize=norm)
        st.dataframe(ct, use_container_width=True)
        df_to_excel_download({"crosstab": ct})

# --- Hypothesis Testing ---
with tab_hypo:
    st.subheader("Hypothesis Generation & Testing")

    left, right = st.columns([1,1])
    with left:
        y_col = st.selectbox("Outcome / Numeric variable (Y)", all_cols, key="y_col")
        x_col = st.selectbox("Grouping / Predictor (X) (optional)", [None] + all_cols, key="x_col")
        auto_test = auto_select_test(df, y_col, x_col) if y_col else "Auto"
        test_name = st.selectbox("Test", ["Auto", "Chi-square", "t-test (independent)", "ANOVA", "F-test (variances)"], index=0, help=TESTS_HELP["Auto"])        
        alpha = st.number_input("Significance level (alpha)", 0.001, 0.2, 0.05, 0.001)
        if st.checkbox("Show suggested test"):
            st.info(f"Suggested: **{auto_test}**")

    with right:
        st.caption("Define hypotheses (optional)")
        h0 = st.text_input("Null hypothesis (H0)", placeholder="e.g., Mean satisfaction is equal across genders.")
        h1 = st.text_input("Alternate hypothesis (H1)", placeholder="e.g., Mean satisfaction differs across genders.")
        if st.button("Generate hypotheses with AI"):
            key = get_user_gemini_key()
            if not key:
                st.warning("Add your Gemini API key in the sidebar to use AI.")
            else:
                prompt = f"""You are a statistician. Given variable types and short samples, propose clear H0 and H1.
Variables: Y={y_col} ({'numeric' if pd.api.types.is_numeric_dtype(df[y_col]) else 'categorical'}) and X={x_col}.
Respond as JSON with keys h0, h1."""
                js = gemini_generate_json(prompt, key, max_output_tokens=256)
                st.write(js)

    if st.button("Run Test", type="primary"):
        chosen = auto_test if test_name == "Auto" else test_name
        result = run_test(df, chosen, y_col, x_col)
        st.write(result)
        if result.get("p_value") is not None:
            decision = "Reject H0" if result["p_value"] < alpha else "Fail to reject H0"
            st.success(f"Decision at α={alpha}: **{decision}** (p={result['p_value']:.4g})")

# --- AI Chat (RAG) ---
with tab_chat:
    st.subheader("Chat with Your Data (RAG + Gemini)")
    if "chat_history" not in st.session_state:
        st.session_state["chat_history"] = []  # list of dicts: {role, content}

    for msg in st.session_state["chat_history"]:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    question = st.chat_input("Ask a question about your data...")
    if question:
        st.session_state["chat_history"].append({"role": "user", "content": question})
        with st.chat_message("user"):
            st.markdown(question)
        chunks = rag_retrieve(question, st.session_state.get("rag_chunks", []), top_k=8)
        key = get_user_gemini_key()
        if not key:
            answer = "Please add your Gemini API key in the sidebar to use AI."
            st.session_state["chat_history"].append({"role": "assistant", "content": answer})
            with st.chat_message("assistant"):
                st.markdown(answer)
        else:
            data_desc = df.describe(include='all').to_csv(errors='ignore') if len(df) > 100 else df.to_csv(index=False)
            ai_mode = "Generate a visualization" if any(w in question.lower() for w in ["plot", "chart", "graph", "visualize"]) else "Answer a question"
            if ai_mode == "Generate a visualization":
                prompt = f"""
You are an expert data visualization analyst. Use the user's question and retrieved context from the dataset to produce code for a Matplotlib graph.
Respond as a single JSON with keys 'answer' and 'code'. The code MUST create a variable `fig` and may use `pd`, `plt`, `np`, and a pandas dataframe named `data`.

Retrieved context (top-k):\n{json.dumps(chunks)[:4000]}

Data sample/summary:\n{data_desc[:4000]}

User question: {question}
"""
            else:
                prompt = f"""
You are a helpful data analyst. Answer based on the dataset and retrieved context. Respond as JSON with keys 'answer' (string) and 'code' (must be null).

Retrieved context (top-k):\n{json.dumps(chunks)[:4000]}

Data sample/summary:\n{data_desc[:4000]}

User question: {question}
"""
            with st.spinner("Thinking with Gemini..."):
                js = gemini_generate_json(prompt, key)

            with st.chat_message("assistant"):
                st.markdown(js.get('answer', '(No answer)'))
                if js.get('code'):
                    try:
                        namespace = {'data': df.copy(), 'pd': pd, 'plt': plt, 'np': np, 'fig': None, 'ax': None, 'df': None}
                        exec(js['code'], namespace)
                        if namespace.get('fig') is not None:
                            st.pyplot(namespace['fig'])
                        if namespace.get('df') is not None:
                            st.dataframe(namespace['df'])
                    except Exception as e:
                        st.error(f"Error executing generated code: {e}")
                        st.code(js['code'], language='python')
            st.session_state["chat_history"].append({"role": "assistant", "content": js.get('answer', '')})

# --- Settings ---
with tab_settings:
    st.subheader("Settings")
    st.write("You are signed in as:")
    st.json({k: user[k] for k in ["name","email","sub"]})

    st.markdown("**Delete stored Gemini key**")
    if st.button("Forget my API key"):
        users = get_users_col()
        users.update_one({"google_sub": user["sub"]}, {"$unset": {"gemini_key": 1}})
        st.success("Key removed.")

    st.caption("All sensitive data is encrypted with Fernet before storage in MongoDB.")

# --- END ---
