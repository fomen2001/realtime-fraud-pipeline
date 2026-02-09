import os
import time
import pandas as pd
import psycopg2
import streamlit as st
import plotly.express as px

# -----------------------------
# Page setup
# -----------------------------
st.set_page_config(page_title="Fraud Monitoring", page_icon="🛡️", layout="wide")

DB_HOST = os.getenv("DB_HOST", "localhost")
DB_PORT = int(os.getenv("DB_PORT", "5432"))
DB_NAME = os.getenv("DB_NAME", "fraud")
DB_USER = os.getenv("DB_USER", "app")
DB_PASS = os.getenv("DB_PASS", "app")

# -----------------------------
# DB helper
# -----------------------------
@st.cache_data(ttl=3, show_spinner=False)
def query(sql: str) -> pd.DataFrame:
    try:
        conn = psycopg2.connect(
            host=DB_HOST, port=DB_PORT, dbname=DB_NAME, user=DB_USER, password=DB_PASS
        )
        try:
            return pd.read_sql(sql, conn)
        finally:
            conn.close()
    except Exception as e:
        st.error(f"❌ Connexion PostgreSQL impossible : {e}")
        return pd.DataFrame()

def safe_datetime(df: pd.DataFrame, col: str = "ts") -> pd.DataFrame:
    if not df.empty and col in df.columns:
        df[col] = pd.to_datetime(df[col], errors="coerce", utc=True)
        # Streamlit + Plotly ok with tz-aware; if needed: df[col] = df[col].dt.tz_convert(None)
    return df

def fmt_eur(x: float) -> str:
    try:
        return f"{x:,.0f} €".replace(",", " ")
    except Exception:
        return f"{x} €"

# -----------------------------
# Sidebar (controls)
# -----------------------------
st.sidebar.title("⚙️ Contrôles")

auto_refresh = st.sidebar.toggle("Auto-refresh", value=True)
refresh_seconds = st.sidebar.slider("Intervalle (secondes)", 2, 15, 3)

st.sidebar.markdown("---")
st.sidebar.subheader("🎛️ Seuil & SLA")

risk_threshold = st.sidebar.slider("Seuil de fraude (risk_score)", 0.50, 0.99, 0.80, 0.01)
sla_window_min = st.sidebar.slider("Fenêtre SLA (minutes)", 1, 15, 5)
sla_alert_rate = st.sidebar.slider("SLA : alerte si taux > (%)", 1.0, 30.0, 8.0, 0.5)

st.sidebar.markdown("---")
st.sidebar.subheader("💰 Hypothèses coût métier")

loss_per_fraud = st.sidebar.number_input("Perte moyenne par fraude (si non stoppée)", min_value=10, max_value=20000, value=250, step=10)
review_cost = st.sidebar.number_input("Coût revue manuelle / transaction flag", min_value=0, max_value=200, value=2, step=1)
false_positive_cost = st.sidebar.number_input("Coût faux positif (friction client, churn, etc.)", min_value=0, max_value=500, value=5, step=1)

# Efficacité opérationnelle (tous les fraudes flag ne sont pas stoppées)
block_efficiency = st.sidebar.slider("Efficacité stop fraude (si flag)", 0.0, 1.0, 0.80, 0.05)

st.sidebar.markdown("---")
st.sidebar.subheader("🧾 Filtres (vue)")

# -----------------------------
# Load data
# -----------------------------
tx = query("SELECT * FROM transactions_enriched ORDER BY ts DESC LIMIT 3000;")
alerts_tbl = query("SELECT * FROM fraud_alerts ORDER BY ts DESC LIMIT 1500;")

tx = safe_datetime(tx, "ts")
alerts_tbl = safe_datetime(alerts_tbl, "ts")

# Guard: risk_score needed for decision analytics
has_risk = (not tx.empty) and ("risk_score" in tx.columns)

# Build filter options
countries = sorted(tx["country"].dropna().unique().tolist()) if not tx.empty and "country" in tx.columns else []
merchants = sorted(tx["merchant"].dropna().unique().tolist()) if not tx.empty and "merchant" in tx.columns else []
categories = sorted(tx["category"].dropna().unique().tolist()) if not tx.empty and "category" in tx.columns else []
customers = sorted(tx["customer_id"].dropna().unique().tolist()) if not tx.empty and "customer_id" in tx.columns else []

country_filter = st.sidebar.multiselect("Pays", options=countries, default=[])
merchant_filter = st.sidebar.multiselect("Marchands", options=merchants, default=[])
category_filter = st.sidebar.multiselect("Catégories", options=categories, default=[])

tx_view = tx.copy()
if country_filter and "country" in tx_view.columns:
    tx_view = tx_view[tx_view["country"].isin(country_filter)]
if merchant_filter and "merchant" in tx_view.columns:
    tx_view = tx_view[tx_view["merchant"].isin(merchant_filter)]
if category_filter and "category" in tx_view.columns:
    tx_view = tx_view[tx_view["category"].isin(category_filter)]

# Flagged transactions based on threshold (business view)
flagged = pd.DataFrame()
if has_risk:
    flagged = tx_view[tx_view["risk_score"] >= risk_threshold].copy()

# -----------------------------
# Header
# -----------------------------
st.title("🛡️ Fraud Monitoring — Temps Réel")
st.caption(
    "Ce dashboard aide à piloter la fraude : volume, score de risque, seuils, hotspots, "
    "impact financier estimé et investigation client."
)

# -----------------------------
# Navigation (tabs)
# -----------------------------
tab_exec, tab_perf, tab_invest = st.tabs(["📌 Executive", "📈 Seuil & Performance", "🔎 Investigation"])

# =========================================================
# TAB 1 — EXECUTIVE
# =========================================================
with tab_exec:
    k1, k2, k3, k4 = st.columns(4)

    tx_count = len(tx_view)
    flagged_count = len(flagged) if not flagged.empty else 0
    alert_rate = (flagged_count / max(tx_count, 1)) * 100

    with k1:
        st.metric("Transactions (vue)", f"{tx_count:,}".replace(",", " "))
    with k2:
        st.metric(f"Flag ≥ seuil ({risk_threshold:.2f})", f"{flagged_count:,}".replace(",", " "))
    with k3:
        st.metric("Taux de risque (vue)", f"{alert_rate:.2f}%")
    with k4:
        level = "FAIBLE"
        if alert_rate >= 8:
            level = "ÉLEVÉ"
        elif alert_rate >= 3:
            level = "MODÉRÉ"
        st.metric("Niveau de risque", level)

    st.markdown("---")

    # -------- SLA / Alerting (5-min window)
    st.subheader("⏱️ SLA & Alerting")
    if not tx_view.empty and "ts" in tx_view.columns and has_risk:
        tmp = tx_view.dropna(subset=["ts"]).copy()
        tmp["minute"] = tmp["ts"].dt.floor("min")

        grp = tmp.groupby("minute").agg(
            tx=("transaction_id", "count") if "transaction_id" in tmp.columns else ("amount", "count"),
            flagged=("risk_score", lambda s: (s >= risk_threshold).sum()),
        ).reset_index()

        grp["alert_rate_pct"] = grp["flagged"] / grp["tx"].clip(lower=1) * 100
        grp = grp.sort_values("minute")

        # rolling window mean (simple proxy)
        grp["sla_rate_pct"] = grp["alert_rate_pct"].rolling(window=sla_window_min, min_periods=1).mean()

        fig_sla = px.line(grp, x="minute", y="sla_rate_pct", title=f"Taux d'alerte moyen (fenêtre {sla_window_min} min)")
        st.plotly_chart(fig_sla, use_container_width=True)

        latest = float(grp["sla_rate_pct"].iloc[-1]) if not grp.empty else 0.0
        if latest > sla_alert_rate:
            st.error(f"🚨 SLA déclenché : taux moyen {latest:.2f}% > seuil {sla_alert_rate:.2f}% (sur {sla_window_min} min)")
            st.toast("SLA: hausse anormale du taux d’alerte — action recommandée", icon="🚨")
        else:
            st.success(f"✅ SLA OK : taux moyen {latest:.2f}% ≤ {sla_alert_rate:.2f}% (sur {sla_window_min} min)")
    else:
        st.info("SLA indisponible : données ou colonne risk_score/ts manquantes.")

    st.markdown("---")

    # -------- Business cost estimation
    st.subheader("💰 Impact métier estimé (probabiliste)")
    with st.expander("Comment ces estimations sont calculées ?", expanded=False):
        st.markdown(
            """
**Sans labels confirmés**, on estime les métriques en traitant `risk_score` comme une probabilité.
- Fraudes attendues dans les transactions flag = somme des risk_score des transactions flag.
- Faux positifs attendus = nb_flag - fraudes_attendues_flag.
- Fraudes manquées attendues (sous seuil) = somme des risk_score sous seuil.

On estime ensuite des coûts:
- **Pertes évitées** ≈ fraudes_attendues_flag × efficacité_blocage × perte_moyenne
- **Coûts** ≈ (revue_manuel × nb_flag) + (faux_positifs × coût_faux_positif)
- **Bénéfice net** = pertes évitées − coûts
"""
        )

    if has_risk and not tx_view.empty:
        # expected counts
        prob = tx_view["risk_score"].clip(0, 1)
        prob_flag = flagged["risk_score"].clip(0, 1) if not flagged.empty else pd.Series([], dtype=float)

        expected_total_fraud = float(prob.sum())
        expected_fraud_flagged = float(prob_flag.sum()) if len(prob_flag) > 0 else 0.0
        expected_fp_flagged = float(flagged_count - expected_fraud_flagged)
        expected_missed_fraud = float(expected_total_fraud - expected_fraud_flagged)

        avoided_losses = expected_fraud_flagged * block_efficiency * float(loss_per_fraud)
        op_costs = (flagged_count * float(review_cost)) + (expected_fp_flagged * float(false_positive_cost))
        net_value = avoided_losses - op_costs

        c1, c2, c3, c4 = st.columns(4)
        with c1:
            st.metric("Fraudes attendues (vue)", f"{expected_total_fraud:.1f}")
        with c2:
            st.metric("Fraudes stoppées attendues", f"{(expected_fraud_flagged*block_efficiency):.1f}")
        with c3:
            st.metric("Pertes évitées (estim.)", fmt_eur(avoided_losses))
        with c4:
            st.metric("Bénéfice net (estim.)", fmt_eur(net_value))

        st.caption(
            f"Coûts inclus : revue ({review_cost}€/flag) + faux positifs ({false_positive_cost}€/FP). "
            f"Efficacité stop fraude = {block_efficiency:.0%}."
        )
    else:
        st.warning("Impact métier indisponible : colonne risk_score manquante ou dataset vide.")

    st.markdown("---")

    # -------- Hotspots & operational tables
    left, right = st.columns([1.2, 1])

    with left:
        st.subheader("📊 Distributions & volume")
        if not tx_view.empty and "amount" in tx_view.columns:
            fig_amount = px.histogram(tx_view, x="amount", nbins=40, title="Distribution des montants (vue filtrée)")
            st.plotly_chart(fig_amount, use_container_width=True)

        if has_risk:
            fig_risk = px.histogram(tx_view, x="risk_score", nbins=30, title="Distribution des scores de risque (vue)")
            st.plotly_chart(fig_risk, use_container_width=True)

        st.subheader("🧾 Dernières transactions")
        if not tx_view.empty:
            cols = [c for c in ["ts", "transaction_id", "customer_id", "amount", "country", "merchant", "category", "risk_score"] if c in tx_view.columns]
            st.dataframe(tx_view.sort_values("ts", ascending=False)[cols].head(30), use_container_width=True)
        else:
            st.info("Aucune transaction à afficher (filtres trop restrictifs ?).")

    with right:
        st.subheader("🎯 Hotspots (où agir)")
        if not flagged.empty:
            if "country" in flagged.columns:
                top_country = flagged["country"].value_counts().head(10).reset_index()
                top_country.columns = ["country", "count"]
                st.plotly_chart(px.bar(top_country, x="country", y="count", title="Top pays à risque (≥ seuil)"), use_container_width=True)

            if "merchant" in flagged.columns:
                top_merchant = flagged["merchant"].value_counts().head(10).reset_index()
                top_merchant.columns = ["merchant", "count"]
                st.plotly_chart(px.bar(top_merchant, x="merchant", y="count", title="Top marchands à risque (≥ seuil)"), use_container_width=True)
        else:
            st.info("Aucune transaction au-dessus du seuil actuel.")

        st.subheader("🚨 Table d’alertes (source)")
        if not alerts_tbl.empty:
            cols_a = [c for c in ["ts", "transaction_id", "severity", "risk_score", "reason", "country", "merchant", "amount"] if c in alerts_tbl.columns]
            st.dataframe(alerts_tbl.sort_values("ts", ascending=False)[cols_a].head(30), use_container_width=True)
        else:
            st.caption("Table fraud_alerts vide (normal si tu n’écris pas dedans).")

# =========================================================
# TAB 2 — THRESHOLD & PERFORMANCE (simulated)
# =========================================================
with tab_perf:
    st.subheader("📈 Precision / Recall estimés selon le seuil (probabiliste)")

    if not has_risk or tx_view.empty:
        st.warning("Impossible de calculer : risk_score manquant ou dataset vide.")
    else:
        # Sweep thresholds
        thresholds = [round(x, 2) for x in list(pd.Series([i/100 for i in range(50, 100)]).values)]
        prob_all = tx_view["risk_score"].clip(0, 1)
        expected_total_fraud = float(prob_all.sum())

        rows = []
        for t in thresholds:
            flag_t = tx_view[tx_view["risk_score"] >= t]
            n_flag = len(flag_t)
            exp_fraud_flag = float(flag_t["risk_score"].clip(0, 1).sum()) if n_flag > 0 else 0.0

            # Expected precision/recall
            precision = exp_fraud_flag / max(n_flag, 1)
            recall = exp_fraud_flag / max(expected_total_fraud, 1e-9)

            # Expected business value
            exp_fp = n_flag - exp_fraud_flag
            avoided = exp_fraud_flag * block_efficiency * float(loss_per_fraud)
            costs = (n_flag * float(review_cost)) + (exp_fp * float(false_positive_cost))
            net = avoided - costs

            rows.append({
                "threshold": t,
                "flagged": n_flag,
                "precision_est": precision,
                "recall_est": recall,
                "net_value_est": net,
                "avoided_losses_est": avoided,
                "costs_est": costs
            })

        df_curve = pd.DataFrame(rows)

        c1, c2 = st.columns([1.2, 1])
        with c1:
            fig_pr = px.line(df_curve, x="threshold", y=["precision_est", "recall_est"], title="Precision & Recall estimés vs seuil")
            st.plotly_chart(fig_pr, use_container_width=True)

        with c2:
            fig_net = px.line(df_curve, x="threshold", y="net_value_est", title="Bénéfice net estimé vs seuil")
            st.plotly_chart(fig_net, use_container_width=True)

        st.caption(
            "⚠️ Ces métriques sont des estimations basées sur risk_score≈probabilité. "
            "Pour de vraies métriques, il faut des labels de fraude confirmée."
        )

        # Show best threshold suggestion (max net value)
        best = df_curve.sort_values("net_value_est", ascending=False).head(1).iloc[0]
        st.success(
            f"🎯 Seuil recommandé (max bénéfice net estimé) : {best['threshold']:.2f} "
            f"(bénéfice ≈ {fmt_eur(best['net_value_est'])}, precision ≈ {best['precision_est']:.2%}, recall ≈ {best['recall_est']:.2%})"
        )

        st.dataframe(df_curve.sort_values("threshold", ascending=True), use_container_width=True, height=320)

# =========================================================
# TAB 3 — INVESTIGATION (drill-down client)
# =========================================================
with tab_invest:
    st.subheader("🔎 Investigation — Drill-down par client")

    if tx.empty or "customer_id" not in tx.columns:
        st.warning("Pas de données client (customer_id manquant).")
    else:
        selected_customer = st.selectbox("Choisir un client", options=customers, index=0 if customers else None)

        cust = tx[tx["customer_id"] == selected_customer].copy()
        cust = cust.sort_values("ts", ascending=True)

        if cust.empty:
            st.info("Aucune transaction pour ce client.")
        else:
            # Summary KPIs for client
            total_c = len(cust)
            flagged_c = len(cust[cust.get("risk_score", 0) >= risk_threshold]) if has_risk else 0

            c1, c2, c3, c4 = st.columns(4)
            with c1:
                st.metric("Transactions client", f"{total_c:,}".replace(",", " "))
            with c2:
                st.metric(f"Flag ≥ seuil ({risk_threshold:.2f})", f"{flagged_c:,}".replace(",", " "))
            with c3:
                if "amount" in cust.columns:
                    st.metric("Montant total", fmt_eur(float(cust["amount"].sum())))
                else:
                    st.metric("Montant total", "N/A")
            with c4:
                if has_risk:
                    st.metric("Score max", f"{float(cust['risk_score'].max()):.2f}")
                else:
                    st.metric("Score max", "N/A")

            st.markdown("---")

            # Timeline charts
            left, right = st.columns([1.3, 1])

            with left:
                if has_risk:
                    fig = px.line(cust.dropna(subset=["ts"]), x="ts", y="risk_score", title="Évolution du score de risque (client)")
                    st.plotly_chart(fig, use_container_width=True)

                if "amount" in cust.columns:
                    fig2 = px.bar(cust.dropna(subset=["ts"]), x="ts", y="amount", title="Montants des transactions (client)")
                    st.plotly_chart(fig2, use_container_width=True)

            with right:
                st.subheader("📍 Profil du client (hotspots)")
                if "country" in cust.columns:
                    top_c = cust["country"].value_counts().head(8).reset_index()
                    top_c.columns = ["country", "count"]
                    st.plotly_chart(px.bar(top_c, x="country", y="count", title="Pays (client)"), use_container_width=True)

                if "merchant" in cust.columns:
                    top_m = cust["merchant"].value_counts().head(8).reset_index()
                    top_m.columns = ["merchant", "count"]
                    st.plotly_chart(px.bar(top_m, x="merchant", y="count", title="Marchands (client)"), use_container_width=True)

            st.subheader("🧾 Transactions client (détaillé)")
            cols = [c for c in ["ts", "transaction_id", "amount", "country", "merchant", "category", "risk_score"] if c in cust.columns]
            st.dataframe(cust.sort_values("ts", ascending=False)[cols].head(200), use_container_width=True)

            if has_risk:
                st.subheader("✅ Actions recommandées (client)")
                high = cust[cust["risk_score"] >= max(0.95, risk_threshold)]
                mid = cust[(cust["risk_score"] >= risk_threshold) & (cust["risk_score"] < 0.95)]
                if len(high) > 0:
                    st.error(f"🚨 {len(high)} transactions très risquées (≥ 0.95) : blocage / investigation immédiate.")
                if len(mid) > 0:
                    st.warning(f"🟠 {len(mid)} transactions au-dessus du seuil : revue manuelle prioritaire.")
                if len(high) == 0 and len(mid) == 0:
                    st.success("🟢 Aucune transaction au-dessus du seuil : surveillance standard.")

# -----------------------------
# Footer + auto refresh
# -----------------------------
st.caption(
    f"🔄 Refresh: {'ON' if auto_refresh else 'OFF'} — {refresh_seconds}s | "
    f"Seuil={risk_threshold:.2f} | SLA {sla_window_min}min > {sla_alert_rate:.2f}%"
)

if auto_refresh:
    time.sleep(refresh_seconds)
    st.rerun()
