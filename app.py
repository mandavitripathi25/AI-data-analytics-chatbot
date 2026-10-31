"""
AI Data Analytics Chatbot (Advanced)
Upload Excel/CSV -> ask questions in plain English -> AI writes SQL, runs it,
draws a chart and explains the insight.

Stack: Python, Streamlit, SQLite (SQL), Pandas, Plotly, Groq API
Run:   streamlit run app.py
"""

import json
import os
import re
import sqlite3

import pandas as pd
import plotly.express as px
import streamlit as st
from groq import Groq

# --------------------------------------------------------------------------
# Page config
# --------------------------------------------------------------------------
st.set_page_config(page_title="AI Data Analytics Chatbot", page_icon="📊", layout="wide")

DEFAULT_MODEL = "llama-3.3-70b-versatile"
MAX_ROWS_SHOWN = 1000
FORBIDDEN_SQL = r"\b(insert|update|delete|drop|alter|create|attach|detach|pragma|vacuum|truncate)\b"


@st.cache_data(ttl=600, show_spinner=False)
def list_chat_models(api_key: str):
    """Ask Groq which models this key can use, keeping only text chat models."""
    try:
        models = Groq(api_key=api_key).models.list().data
    except Exception:
        return []
    skip = ("whisper", "tts", "guard", "orpheus", "embed", "playai", "distil-whisper")
    ids = [m.id for m in models if not any(s in m.id.lower() for s in skip)]
    return sorted(ids)


# --------------------------------------------------------------------------
# Data loading helpers
# --------------------------------------------------------------------------
def clean_name(name: str) -> str:
    """Make a safe SQL identifier: lowercase, underscores, no symbols."""
    name = re.sub(r"[^0-9a-zA-Z]+", "_", str(name).strip()).strip("_").lower()
    if not name:
        name = "col"
    if name[0].isdigit():
        name = "t_" + name
    return name


def load_files(files):
    """Load uploaded CSV/Excel files into an in-memory SQLite DB. Returns (conn, tables)."""
    conn = sqlite3.connect(":memory:", check_same_thread=False)
    tables = {}
    for f in files:
        base = clean_name(os.path.splitext(f.name)[0])
        if f.name.lower().endswith(".csv"):
            frames = {base: pd.read_csv(f)}
        else:
            sheets = pd.read_excel(f, sheet_name=None)  # all sheets
            frames = {
                (base if len(sheets) == 1 else f"{base}_{clean_name(s)}"): df
                for s, df in sheets.items()
            }
        for tname, df in frames.items():
            df = df.dropna(how="all").dropna(axis=1, how="all")
            df.columns = [clean_name(c) for c in df.columns]
            # de-duplicate column names
            seen, cols = {}, []
            for c in df.columns:
                seen[c] = seen.get(c, 0) + 1
                cols.append(c if seen[c] == 1 else f"{c}_{seen[c]}")
            df.columns = cols
            # try to parse date-like text columns
            for c in df.columns:
                if df[c].dtype == "object":
                    try:
                        parsed = pd.to_datetime(df[c], errors="raise")
                        df[c] = parsed.dt.strftime("%Y-%m-%d")
                    except Exception:
                        pass
            df.to_sql(tname, conn, index=False, if_exists="replace")
            tables[tname] = df
    conn.execute("PRAGMA query_only = ON")  # extra safety: DB becomes read-only
    return conn, tables


def build_schema_text(tables: dict) -> str:
    parts = []
    for tname, df in tables.items():
        cols = ", ".join(f"{c} ({str(t)})" for c, t in df.dtypes.items())
        sample = df.head(3).to_csv(index=False)
        parts.append(f"TABLE {tname} ({len(df)} rows)\nColumns: {cols}\nSample rows:\n{sample}")
    return "\n".join(parts)


# --------------------------------------------------------------------------
# SQL safety + execution
# --------------------------------------------------------------------------
def run_sql(conn, sql: str) -> pd.DataFrame:
    s = sql.strip().rstrip(";").strip()
    if not re.match(r"^(select|with)\b", s, re.I):
        raise ValueError("Only SELECT queries are allowed.")
    if ";" in s:
        raise ValueError("Multiple statements are not allowed.")
    if re.search(FORBIDDEN_SQL, s, re.I):
        raise ValueError("Query contains a forbidden keyword.")
    return pd.read_sql_query(s, conn)


# --------------------------------------------------------------------------
# LLM calls (Groq)
# --------------------------------------------------------------------------
SQL_SYSTEM_PROMPT = """You are an expert data analyst who writes SQLite queries.
You are given the database schema. Convert the user's question into ONE SQLite SELECT query.

Rules:
- Use only tables and columns that exist in the schema. Column names are lowercase.
- SQLite dialect only (use strftime for dates, no ILIKE, no DATE_TRUNC).
- Only SELECT / WITH queries. Never modify data.
- Use clear column aliases. Add ORDER BY and LIMIT (max 100) when it makes sense.
- If the question is vague, make a reasonable assumption and state it in "explanation".

Respond with ONLY a JSON object:
{
  "sql": "<the query>",
  "chart": {"type": "bar|line|pie|scatter|histogram|none",
            "x": "<result column>", "y": "<result column or null>",
            "color": "<result column or null>", "title": "<short title>"},
  "explanation": "<one sentence about what the query does>"
}
Chart guidance: bar for category comparison, line for trends over time,
pie for share of total (<=8 slices), scatter for two numeric columns,
histogram for distribution, none if a chart makes no sense.
x / y / color must be column names (aliases) from your SELECT result."""


def extract_json(text: str) -> dict:
    """Parse JSON even if the model adds <think> blocks or extra text around it."""
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.S).strip()
    try:
        return json.loads(text)
    except Exception:
        match = re.search(r"\{.*\}", text, re.S)
        if not match:
            raise ValueError("Model did not return valid JSON.")
        return json.loads(match.group(0))


def call_llm_json(client, model, system, user):
    messages = [
        {"role": "system", "content": system},
        {"role": "user", "content": user},
    ]
    try:
        resp = client.chat.completions.create(
            model=model,
            temperature=0,
            response_format={"type": "json_object"},
            messages=messages,
        )
    except Exception as e:
        # some models don't support JSON mode -> retry without it
        if "response_format" in str(e) or "json" in str(e).lower():
            resp = client.chat.completions.create(
                model=model, temperature=0, messages=messages
            )
        else:
            raise
    return extract_json(resp.choices[0].message.content)


def generate_sql(client, model, schema, question, history, error=None, bad_sql=None):
    user = f"DATABASE SCHEMA:\n{schema}\n\n"
    if history:
        user += "RECENT QUESTIONS (for context):\n" + "\n".join(f"- {q}" for q in history) + "\n\n"
    user += f"QUESTION: {question}"
    if error:
        user += (
            f"\n\nYour previous query failed.\nQuery: {bad_sql}\nError: {error}\n"
            "Fix it and return the corrected JSON."
        )
    return call_llm_json(client, model, SQL_SYSTEM_PROMPT, user)


def generate_insights(client, model, question, sql, df):
    preview = df.head(30).to_csv(index=False)
    prompt = f"""Question: {question}
SQL used: {sql}
Result ({len(df)} rows, first 30 shown):
{preview}

Write a short business-style insight for a non-technical reader:
- 2 to 4 bullet points, include concrete numbers from the result
- point out the top/bottom values, trends or anything unusual
- end with one suggested follow-up question
Do not invent numbers that are not in the result."""
    resp = client.chat.completions.create(
        model=model,
        temperature=0.3,
        messages=[
            {"role": "system", "content": "You are a clear, concise data analyst."},
            {"role": "user", "content": prompt},
        ],
    )
    return resp.choices[0].message.content


# --------------------------------------------------------------------------
# Charting
# --------------------------------------------------------------------------
def make_chart(df: pd.DataFrame, spec: dict):
    if df.empty or not spec or spec.get("type") in (None, "none"):
        return None
    cols = list(df.columns)
    num_cols = df.select_dtypes("number").columns.tolist()

    x = spec.get("x") if spec.get("x") in cols else None
    y = spec.get("y") if spec.get("y") in cols else None
    color = spec.get("color") if spec.get("color") in cols else None
    title = spec.get("title") or ""
    ctype = spec["type"]

    # sensible fallbacks if the model picked missing columns
    if x is None:
        x = cols[0]
    if y is None and ctype != "histogram":
        y = next((c for c in num_cols if c != x), None)

    try:
        if ctype == "bar":
            return px.bar(df, x=x, y=y, color=color, title=title)
        if ctype == "line":
            return px.line(df, x=x, y=y, color=color, markers=True, title=title)
        if ctype == "pie":
            return px.pie(df, names=x, values=y, title=title)
        if ctype == "scatter":
            return px.scatter(df, x=x, y=y, color=color, title=title)
        if ctype == "histogram":
            return px.histogram(df, x=x, title=title)
    except Exception:
        return None
    return None


# --------------------------------------------------------------------------
# UI
# --------------------------------------------------------------------------
st.title("📊 AI Data Analytics Chatbot")
st.caption("Upload data → ask in plain English → get SQL, charts and insights.")

with st.sidebar:
    st.header("⚙️ Settings")
    api_key = st.text_input("Groq API key", type="password", value=os.getenv("GROQ_API_KEY", ""))
    available = list_chat_models(api_key) if api_key else []
    if available:
        preferred = [m for m in available if "llama-3.3-70b" in m or "llama-4" in m or "gpt-oss" in m]
        default_idx = available.index(preferred[0]) if preferred else 0
        model = st.selectbox("Model (live list from Groq)", available, index=default_idx)
    else:
        model = st.text_input("Model", value=DEFAULT_MODEL)
        st.caption("Enter a valid API key to load the available models.")
    st.caption("Get a free key at console.groq.com")
    st.divider()
    uploaded = st.file_uploader(
        "Upload Excel / CSV", type=["csv", "xlsx", "xls"], accept_multiple_files=True
    )
    if st.button("🗑️ Clear chat"):
        st.session_state.messages = []
        st.rerun()

if "messages" not in st.session_state:
    st.session_state.messages = []

if not uploaded:
    st.info("👈 Upload one or more CSV / Excel files from the sidebar to begin.")
    st.stop()

# (re)load the database only when the uploaded files change
files_key = tuple((f.name, f.size) for f in uploaded)
if st.session_state.get("files_key") != files_key:
    try:
        conn, tables = load_files(uploaded)
    except Exception as e:
        st.error(f"Could not read the files: {e}")
        st.stop()
    st.session_state.update(
        files_key=files_key, conn=conn, tables=tables, messages=[]
    )

conn = st.session_state.conn
tables = st.session_state.tables
schema = build_schema_text(tables)

with st.expander("📁 Loaded tables & preview", expanded=False):
    for tname, df in tables.items():
        st.markdown(f"**{tname}** — {len(df)} rows × {len(df.columns)} columns")
        st.dataframe(df.head(10), use_container_width=True)

# example prompts
st.markdown("**Try asking:** *Top 5 categories by total sales* · *Monthly trend of revenue* · "
            "*Which region has the lowest average profit?*")


def render_message(i: int, m: dict):
    with st.chat_message(m["role"]):
        st.markdown(m["content"])
        if m.get("sql"):
            with st.expander("🧾 Generated SQL"):
                st.code(m["sql"], language="sql")
        if m.get("df") is not None:
            st.dataframe(m["df"].head(MAX_ROWS_SHOWN), use_container_width=True)
            st.download_button(
                "⬇️ Download result (CSV)",
                m["df"].to_csv(index=False).encode(),
                file_name=f"result_{i}.csv",
                key=f"dl_{i}",
            )
        if m.get("fig") is not None:
            st.plotly_chart(m["fig"], use_container_width=True, key=f"fig_{i}")
        if m.get("insight"):
            st.markdown("#### 💡 Insights")
            st.markdown(m["insight"])


for i, m in enumerate(st.session_state.messages):
    render_message(i, m)

question = st.chat_input("Ask a question about your data...")

if question:
    if not api_key:
        st.warning("Please enter your Groq API key in the sidebar.")
        st.stop()

    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)

    client = Groq(api_key=api_key)
    history = [m["content"] for m in st.session_state.messages if m["role"] == "user"][-4:-1]

    with st.chat_message("assistant"):
        try:
            with st.spinner("Thinking and writing SQL..."):
                plan = generate_sql(client, model, schema, question, history)
                sql = plan.get("sql", "")
                try:
                    result = run_sql(conn, sql)
                except Exception as first_err:
                    # self-correction: give the error back to the model once
                    plan = generate_sql(
                        client, model, schema, question, history,
                        error=str(first_err), bad_sql=sql,
                    )
                    sql = plan.get("sql", "")
                    result = run_sql(conn, sql)

            fig = make_chart(result, plan.get("chart", {}))

            insight = None
            if not result.empty:
                with st.spinner("Generating insights..."):
                    insight = generate_insights(client, model, question, sql, result)

            msg = {
                "role": "assistant",
                "content": plan.get("explanation", "Here is the result:")
                if not result.empty
                else "The query ran fine but returned no rows. Try rephrasing the question.",
                "sql": sql,
                "df": result,
                "fig": fig,
                "insight": insight,
            }
        except Exception as e:
            msg = {
                "role": "assistant",
                "content": f"⚠️ Sorry, I could not answer that: `{e}`\n\nTry rephrasing, "
                           "or mention the exact column names.",
            }

        st.session_state.messages.append(msg)
        render_message(len(st.session_state.messages) - 1, msg)