import os
import time
import pandas as pd
import psycopg2
import streamlit as st
import plotly.express as px

st.set_page_config(page_title="Realtime Fraud Dashboard", layout="wide")

DB_HOST = os.getenv("DB_HOST", "localhost")
DB_PORT = int(os.getenv("DB_PORT", "5432"))
DB_NAME = os.getenv("DB_NAME", "fraud")
DB_USER = os.getenv("DB_USER", "app")
DB_PASS = os.getenv("DB_PASS", "app")

@st.cache_data(ttl=3)
def query(sql: str) -> pd.DataFrame:
    conn = psycopg2.connect(
        host=DB_HOST, port=DB_PORT, dbname=DB_NAME, user=DB_USER, password=DB_PASS
    )
    try:
        df = pd.read_sql(sql, conn)
        return df
    finally:
        conn.close()

st.title("📡 Realtime Fraud Dashboard")

colA, colB, colC = st.columns(3)

tx = query("SELECT * FROM transactions_enriched ORDER BY ts DESC LIMIT 500;")
alerts = query("SELECT * FROM fraud_alerts ORDER BY ts DESC LIMIT 200;")

with colA:
    st.metric("Transactions (last 500)", len(tx))
with colB:
    st.metric("Alerts (last 200)", len(alerts))
with colC:
    rate = (len(alerts) / max(len(tx), 1)) * 100
    st.metric("Alert rate", f"{rate:.2f}%")

left, right = st.columns(2)

with left:
    if len(tx) > 0:
        fig = px.histogram(tx, x="amount", nbins=30, title="Amounts distribution")
        st.plotly_chart(fig, use_container_width=True)
    st.subheader("Latest transactions")
    st.dataframe(tx.head(30), use_container_width=True)

with right:
    st.subheader("Fraud alerts")
    st.dataframe(alerts.head(30), use_container_width=True)

    if len(alerts) > 0:
        fig2 = px.histogram(alerts, x="severity", title="Alerts by severity")
        st.plotly_chart(fig2, use_container_width=True)

st.caption("Auto-refresh every ~3 seconds")
time.sleep(3)
st.rerun()
