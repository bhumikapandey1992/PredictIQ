import os

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import requests
import streamlit as st

API_BASE = os.environ.get("API_BASE", "http://localhost:8000")

st.set_page_config(page_title="PredictIQ", page_icon="📊", layout="wide")

st.markdown("""
<style>
    .block-container { padding-top: 1.5rem; }
    .stMetric { background: #1e1e2e; border-radius: 8px; padding: 0.75rem; }
</style>
""", unsafe_allow_html=True)

st.title("PredictIQ — Predictive Analytics Dashboard")

# ── Sidebar ──────────────────────────────────────────────────────────────────
with st.sidebar:
    st.header("Configuration")
    domain = st.selectbox("Domain", ["stocks", "sales", "healthcare"])
    model_type = st.radio("Model", ["xgboost", "lightgbm"])

    if domain == "stocks":
        ticker = st.text_input("Ticker Symbol", "AAPL")
        period = st.selectbox("Historical Period", ["1y", "2y", "5y"])

    st.divider()

    if st.button("Ingest Data", use_container_width=True, type="primary"):
        params = {}
        if domain == "stocks":
            params = {"ticker": ticker, "period": period}
        with st.spinner("Fetching data..."):
            r = requests.post(f"{API_BASE}/ingest/{domain}", params=params)
        if r.ok:
            payload = r.json()
            st.session_state[f"{domain}_ingested"] = True
            st.session_state[f"{domain}_payload"] = payload
            st.success(f"Loaded {payload['rows']:,} rows")
        else:
            st.error("Ingestion failed")

    if st.session_state.get(f"{domain}_ingested"):
        if st.button("Train Model", use_container_width=True):
            with st.spinner(f"Training {model_type}..."):
                r = requests.post(
                    f"{API_BASE}/train/{domain}", params={"model_type": model_type}
                )
            if r.ok:
                result = r.json()
                st.session_state[f"{domain}_trained"] = True
                st.session_state[f"{domain}_train_result"] = result
                st.success("Model trained!")
            else:
                st.error("Training failed")

# ── Tabs ─────────────────────────────────────────────────────────────────────
tab1, tab2, tab3 = st.tabs(["📂 Data Explorer", "📈 Model Performance", "🔮 Predict"])

# ── TAB 1: Data Explorer ─────────────────────────────────────────────────────
with tab1:
    if not st.session_state.get(f"{domain}_ingested"):
        st.info("Select a domain and click **Ingest Data** in the sidebar to begin.")
    else:
        payload = st.session_state[f"{domain}_payload"]
        df = pd.DataFrame(payload["records"])

        st.markdown(f"**{payload['rows']:,} rows · {len(payload['columns'])} columns**")
        st.dataframe(df, use_container_width=True, height=250)

        st.subheader("Summary Statistics")
        stats_df = pd.DataFrame(payload["stats"]).T
        st.dataframe(stats_df.style.format("{:.3f}", na_rep="-"), use_container_width=True)

        st.subheader("Charts")

        if domain == "stocks":
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=df["date"], y=df["close"], name="Close",
                line=dict(color="#4f46e5", width=1.5),
            ))
            fig.update_layout(title="Stock Close Price", xaxis_title="Date",
                              yaxis_title="Price (USD)", height=350)
            st.plotly_chart(fig, use_container_width=True)

            fig2 = px.bar(df.tail(60), x="date", y="volume",
                          title="Volume — Last 60 Sessions",
                          color_discrete_sequence=["#7c3aed"])
            fig2.update_layout(height=280)
            st.plotly_chart(fig2, use_container_width=True)

        elif domain == "sales":
            df["date"] = pd.to_datetime(df["date"])
            fig = px.line(df, x="date", y="sales", title="Daily Sales Over Time",
                          color_discrete_sequence=["#4f46e5"])
            fig.update_layout(height=320)
            st.plotly_chart(fig, use_container_width=True)

            col1, col2 = st.columns(2)
            with col1:
                fig2 = px.box(df, x=df["promo"].map({0: "No Promo", 1: "Promo"}),
                              y="sales", title="Sales by Promotion",
                              color_discrete_sequence=["#4f46e5", "#22c55e"])
                st.plotly_chart(fig2, use_container_width=True)
            with col2:
                fig3 = px.scatter(df, x="price", y="sales", color="promo",
                                  title="Price vs Sales",
                                  color_continuous_scale="Viridis")
                st.plotly_chart(fig3, use_container_width=True)

        elif domain == "healthcare":
            col1, col2 = st.columns(2)
            with col1:
                counts = df["target"].map({1: "Benign", 0: "Malignant"}).value_counts()
                fig = px.pie(values=counts.values, names=counts.index,
                             title="Diagnosis Distribution",
                             color_discrete_sequence=["#4f46e5", "#ef4444"])
                st.plotly_chart(fig, use_container_width=True)
            with col2:
                top_cols = [c for c in df.columns if c != "target"][:8]
                corr = df[top_cols].corr()
                fig2 = px.imshow(corr, title="Feature Correlation (top 8)",
                                 color_continuous_scale="RdBu", zmin=-1, zmax=1)
                st.plotly_chart(fig2, use_container_width=True)

# ── TAB 2: Model Performance ──────────────────────────────────────────────────
with tab2:
    if not st.session_state.get(f"{domain}_trained"):
        st.info("Train a model from the sidebar to see performance metrics.")
    else:
        result = st.session_state[f"{domain}_train_result"]
        metrics = result["metrics"]
        importance = result["importance"]

        st.subheader("Metrics")
        cols = st.columns(len(metrics))
        for col, (k, v) in zip(cols, metrics.items()):
            col.metric(k.replace("_", " ").upper(), v)

        st.subheader("Feature Importance")
        imp_df = pd.DataFrame(
            list(importance.items()), columns=["Feature", "Importance"]
        ).sort_values("Importance")
        fig = px.bar(
            imp_df, x="Importance", y="Feature", orientation="h",
            title=f"Feature Importance — {model_type.upper()}",
            color="Importance", color_continuous_scale="Viridis",
        )
        fig.update_layout(height=max(300, len(imp_df) * 28), showlegend=False)
        st.plotly_chart(fig, use_container_width=True)

# ── TAB 3: Predict ────────────────────────────────────────────────────────────
with tab3:
    if not st.session_state.get(f"{domain}_trained"):
        st.info("Train a model first.")
    else:
        st.subheader("Make a Prediction")
        feature_inputs: dict = {}

        if domain == "stocks":
            col1, col2 = st.columns(2)
            with col1:
                feature_inputs["return_1d"] = st.slider("1-Day Return", -0.15, 0.15, 0.0, 0.001, format="%.3f")
                feature_inputs["return_5d"] = st.slider("5-Day Return", -0.25, 0.25, 0.0, 0.001, format="%.3f")
                feature_inputs["rsi"] = st.slider("RSI (14)", 0.0, 100.0, 50.0, 0.5)
            with col2:
                feature_inputs["ma_ratio"] = st.slider("MA Ratio (10/20)", 0.90, 1.10, 1.00, 0.001, format="%.3f")
                feature_inputs["vol_change"] = st.slider("Volume Change", -1.0, 2.0, 0.0, 0.01, format="%.2f")

        elif domain == "sales":
            col1, col2 = st.columns(2)
            with col1:
                feature_inputs["store_id"] = st.number_input("Store ID", 1, 5, 1)
                feature_inputs["product_id"] = st.number_input("Product ID", 1, 10, 1)
                feature_inputs["price"] = st.slider("Price ($)", 5.0, 100.0, 50.0, 0.5)
                feature_inputs["promo"] = int(st.checkbox("Promotion Active"))
                feature_inputs["temperature"] = st.slider("Temperature (°C)", 10.0, 35.0, 22.0)
                feature_inputs["day_of_week"] = st.selectbox(
                    "Day of Week", range(7),
                    format_func=lambda x: ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"][x],
                )
            with col2:
                feature_inputs["month"] = st.selectbox("Month", range(1, 13))
                feature_inputs["day_of_year"] = st.number_input("Day of Year", 1, 365, 180)
                feature_inputs["lag_1"] = st.number_input("Yesterday's Sales", 1, 200, 50)
                feature_inputs["lag_7"] = st.number_input("Sales 7 Days Ago", 1, 200, 50)
                feature_inputs["rolling_7"] = st.number_input("7-Day Rolling Mean", 1.0, 200.0, 50.0)

        elif domain == "healthcare":
            importance = st.session_state[f"{domain}_train_result"]["importance"]
            top_features = list(importance.keys())[:10]
            st.caption("Inputs for top 10 features by importance (remaining use value 0).")

            feature_ranges = {
                "mean radius": (6.0, 28.0, 14.0),
                "mean texture": (9.0, 40.0, 19.0),
                "mean perimeter": (43.0, 190.0, 92.0),
                "mean area": (143.0, 2501.0, 655.0),
                "mean smoothness": (0.05, 0.17, 0.10),
                "mean compactness": (0.02, 0.35, 0.10),
                "mean concavity": (0.0, 0.43, 0.09),
                "mean concave points": (0.0, 0.20, 0.05),
                "mean symmetry": (0.11, 0.30, 0.18),
                "mean fractal dimension": (0.05, 0.10, 0.06),
                "worst radius": (7.0, 36.0, 16.0),
                "worst texture": (12.0, 50.0, 26.0),
                "worst perimeter": (50.0, 252.0, 107.0),
                "worst area": (185.0, 4254.0, 881.0),
                "worst smoothness": (0.07, 0.22, 0.13),
                "worst compactness": (0.03, 1.06, 0.25),
                "worst concavity": (0.0, 1.25, 0.27),
                "worst concave points": (0.0, 0.29, 0.11),
                "worst symmetry": (0.16, 0.66, 0.29),
                "worst fractal dimension": (0.06, 0.21, 0.08),
            }

            cols = st.columns(2)
            for i, feat in enumerate(top_features):
                lo, hi, default = feature_ranges.get(feat, (0.0, 1.0, 0.5))
                with cols[i % 2]:
                    feature_inputs[feat] = st.slider(feat, float(lo), float(hi), float(default))

        if st.button("Run Prediction", type="primary"):
            r = requests.post(
                f"{API_BASE}/predict/{domain}",
                json={"features": feature_inputs},
            )
            if r.ok:
                res = r.json()
                st.divider()

                if "label" in res:
                    color = (
                        "green" if res["label"] in ("Up", "Benign") else "red"
                    )
                    st.markdown(
                        f"### Prediction: <span style='color:{color}'>{res['label']}</span>",
                        unsafe_allow_html=True,
                    )
                    if "probabilities" in res:
                        classes = (
                            ["Down", "Up"] if domain == "stocks"
                            else ["Malignant", "Benign"]
                        )
                        prob_df = pd.DataFrame({
                            "Class": classes,
                            "Probability": res["probabilities"],
                        })
                        fig = px.bar(
                            prob_df, x="Class", y="Probability", color="Class",
                            title="Prediction Probabilities",
                            color_discrete_map={
                                "Up": "#22c55e", "Down": "#ef4444",
                                "Benign": "#22c55e", "Malignant": "#ef4444",
                            },
                        )
                        fig.update_layout(showlegend=False, yaxis_range=[0, 1])
                        st.plotly_chart(fig, use_container_width=True)
                else:
                    st.metric("Predicted Sales", f"{res['prediction']:.1f} units")
            else:
                st.error("Prediction failed — check that the backend is running.")
