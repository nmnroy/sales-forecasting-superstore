"""
Sales Forecasting & Demand Intelligence Dashboard
Superstore Sales Dataset — Streamlit deployment

Run locally:  streamlit run app.py
"""

import warnings
warnings.filterwarnings('ignore')

import pandas as pd
import numpy as np
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from statsmodels.tsa.statespace.sarimax import SARIMAX
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA

st.set_page_config(page_title="Sales Forecasting & Demand Intelligence", layout="wide")

# ---------------------------------------------------------------------------
# Data loading & feature engineering (cached so it only runs once per session)
# ---------------------------------------------------------------------------

@st.cache_data
def load_data():
    df = pd.read_csv("train.csv", encoding="latin1")
    df["Order Date"] = pd.to_datetime(df["Order Date"], dayfirst=True)
    df["Ship Date"] = pd.to_datetime(df["Ship Date"], dayfirst=True)
    df["Year"] = df["Order Date"].dt.year
    df["Month"] = df["Order Date"].dt.month
    df["Quarter"] = df["Order Date"].dt.quarter
    return df


@st.cache_data
def get_monthly_series(df, category=None, region=None):
    d = df.copy()
    if category and category != "All":
        d = d[d["Category"] == category]
    if region and region != "All":
        d = d[d["Region"] == region]
    s = d.set_index("Order Date").resample("MS")["Sales"].sum()
    if len(s) > 0:
        s = s.reindex(pd.date_range(s.index.min(), s.index.max(), freq="MS"), fill_value=0)
    return s


@st.cache_data
def run_sarima_forecast(series, horizon):
    train = series.iloc[:-3] if len(series) > 15 else series.iloc[:-2]
    test = series.iloc[-3:] if len(series) > 15 else series.iloc[-2:]
    try:
        model = SARIMAX(train, order=(1, 1, 1), seasonal_order=(1, 1, 1, 12),
                         enforce_stationarity=False, enforce_invertibility=False)
        fit = model.fit(disp=False)
        test_pred = fit.get_forecast(steps=len(test)).predicted_mean
        mae = mean_absolute_error(test, test_pred)
        rmse = np.sqrt(mean_squared_error(test, test_pred))

        full_fit = SARIMAX(series, order=(1, 1, 1), seasonal_order=(1, 1, 1, 12),
                            enforce_stationarity=False, enforce_invertibility=False).fit(disp=False)
        fc = full_fit.get_forecast(steps=horizon)
        pred = fc.predicted_mean
        ci = fc.conf_int()
        return pred, ci, mae, rmse
    except Exception:
        return None, None, None, None


@st.cache_data
def get_anomalies(df):
    weekly = df.set_index("Order Date").resample("W")["Sales"].sum().reset_index()
    weekly.columns = ["week", "sales"]

    iso = IsolationForest(contamination=0.05, random_state=42)
    weekly["iso_anomaly"] = iso.fit_predict(weekly[["sales"]])

    weekly["roll_mean"] = weekly["sales"].rolling(8, min_periods=1).mean()
    weekly["roll_std"] = weekly["sales"].rolling(8, min_periods=1).std()
    weekly["zscore"] = (weekly["sales"] - weekly["roll_mean"]) / weekly["roll_std"]
    weekly["z_anomaly"] = weekly["zscore"].abs() > 2
    return weekly


@st.cache_data
def get_clusters(df):
    sub_monthly = df.groupby(["Sub-Category", df["Order Date"].dt.to_period("M")])["Sales"].sum().reset_index()
    sub_monthly.columns = ["SubCategory", "Period", "Sales"]

    rows = []
    for sc, g in sub_monthly.groupby("SubCategory"):
        g = g.sort_values("Period")
        total_vol = g["Sales"].sum()
        volatility = g["Sales"].std()
        g["Year"] = g["Period"].dt.year
        yearly = g.groupby("Year")["Sales"].sum()
        growth = (yearly.iloc[-1] - yearly.iloc[0]) / yearly.iloc[0] if len(yearly) >= 2 else 0
        rows.append({"SubCategory": sc, "TotalSales": total_vol, "Volatility": volatility, "GrowthRate": growth})

    feat_df = pd.DataFrame(rows)
    aov = df.groupby("Sub-Category")["Sales"].mean().reset_index()
    aov.columns = ["SubCategory", "AvgOrderValue"]
    feat_df = feat_df.merge(aov, on="SubCategory")

    X = feat_df[["TotalSales", "Volatility", "GrowthRate", "AvgOrderValue"]]
    Xs = StandardScaler().fit_transform(X)

    km = KMeans(n_clusters=4, random_state=42, n_init=10)
    feat_df["Cluster"] = km.fit_predict(Xs)

    cluster_means = feat_df.groupby("Cluster")[["TotalSales", "Volatility", "GrowthRate", "AvgOrderValue"]].mean()
    vol_median = cluster_means["TotalSales"].median()
    labels = {}
    for c in cluster_means.index:
        row = cluster_means.loc[c]
        if row["GrowthRate"] < 0:
            labels[c] = "Declining Demand"
        elif row["GrowthRate"] > 1.5:
            labels[c] = "Growing Demand"
        elif row["TotalSales"] >= vol_median:
            labels[c] = "High Volume, Stable Demand"
        else:
            labels[c] = "Low Volume, High Volatility"
    feat_df["ClusterLabel"] = feat_df["Cluster"].map(labels)

    pca = PCA(n_components=2)
    pcs = pca.fit_transform(Xs)
    feat_df["PC1"] = pcs[:, 0]
    feat_df["PC2"] = pcs[:, 1]
    return feat_df


df = load_data()

st.sidebar.title("📊 Sales Intelligence")
page = st.sidebar.radio("Navigate", [
    "1. Sales Overview",
    "2. Forecast Explorer",
    "3. Anomaly Report",
    "4. Product Demand Segments",
])

# ---------------------------------------------------------------------------
# PAGE 1 — Sales Overview
# ---------------------------------------------------------------------------
if page == "1. Sales Overview":
    st.title("Sales Overview Dashboard")

    col1, col2 = st.columns(2)
    with col1:
        region_filter = st.selectbox("Filter by Region", ["All"] + sorted(df["Region"].unique().tolist()))
    with col2:
        category_filter = st.selectbox("Filter by Category", ["All"] + sorted(df["Category"].unique().tolist()))

    filtered = df.copy()
    if region_filter != "All":
        filtered = filtered[filtered["Region"] == region_filter]
    if category_filter != "All":
        filtered = filtered[filtered["Category"] == category_filter]

    total_sales = filtered["Sales"].sum()
    total_orders = filtered["Order ID"].nunique()
    avg_order = filtered.groupby("Order ID")["Sales"].sum().mean()

    m1, m2, m3 = st.columns(3)
    m1.metric("Total Sales", f"${total_sales:,.0f}")
    m2.metric("Total Orders", f"{total_orders:,}")
    m3.metric("Avg Order Value", f"${avg_order:,.2f}")

    st.subheader("Total Sales by Year")
    yearly = filtered.groupby("Year")["Sales"].sum().reset_index()
    fig = px.bar(yearly, x="Year", y="Sales", color_discrete_sequence=["#2563eb"])
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("Monthly Sales Trend")
    monthly = filtered.set_index("Order Date").resample("MS")["Sales"].sum().reset_index()
    fig2 = px.line(monthly, x="Order Date", y="Sales", markers=True)
    fig2.update_traces(line_color="#2563eb")
    st.plotly_chart(fig2, use_container_width=True)

    col3, col4 = st.columns(2)
    with col3:
        st.subheader("Sales by Region")
        reg_sales = filtered.groupby("Region")["Sales"].sum().reset_index()
        fig3 = px.pie(reg_sales, names="Region", values="Sales", hole=0.4)
        st.plotly_chart(fig3, use_container_width=True)
    with col4:
        st.subheader("Sales by Category")
        cat_sales = filtered.groupby("Category")["Sales"].sum().reset_index()
        fig4 = px.bar(cat_sales, x="Category", y="Sales", color="Category",
                      color_discrete_sequence=["#2563eb", "#f97316", "#16a34a"])
        st.plotly_chart(fig4, use_container_width=True)

# ---------------------------------------------------------------------------
# PAGE 2 — Forecast Explorer
# ---------------------------------------------------------------------------
elif page == "2. Forecast Explorer":
    st.title("Forecast Explorer")
    st.caption("Forecasts generated with SARIMA (the model recommended for production — see notebook Task 3).")

    col1, col2 = st.columns(2)
    with col1:
        dim = st.selectbox("Select dimension", ["Category", "Region"])
    with col2:
        if dim == "Category":
            options = sorted(df["Category"].unique().tolist())
        else:
            options = sorted(df["Region"].unique().tolist())
        selection = st.selectbox(f"Select {dim}", options)

    horizon = st.slider("Forecast horizon (months ahead)", 1, 3, 3)

    if dim == "Category":
        series = get_monthly_series(df, category=selection)
    else:
        series = get_monthly_series(df, region=selection)

    with st.spinner("Fitting SARIMA model..."):
        pred, ci, mae, rmse = run_sarima_forecast(series, horizon)

    if pred is not None:
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=series.index, y=series.values, mode="lines", name="Actual",
                                  line=dict(color="#2563eb", width=2)))
        fig.add_trace(go.Scatter(x=pred.index, y=pred.values, mode="lines+markers", name="Forecast",
                                  line=dict(color="#f97316", width=2, dash="dash")))
        fig.add_trace(go.Scatter(x=list(ci.index) + list(ci.index[::-1]),
                                  y=list(ci.iloc[:, 1]) + list(ci.iloc[:, 0][::-1]),
                                  fill="toself", fillcolor="rgba(249,115,22,0.15)",
                                  line=dict(color="rgba(0,0,0,0)"), name="Confidence Interval"))
        fig.update_layout(title=f"{selection} — {horizon}-Month Forecast", xaxis_title="Date", yaxis_title="Sales ($)")
        st.plotly_chart(fig, use_container_width=True)

        st.subheader("Forecast values")
        fc_table = pd.DataFrame({
            "Month": pred.index.strftime("%b %Y"),
            "Forecast ($)": pred.values.round(2),
            "Lower Bound": ci.iloc[:, 0].values.round(2),
            "Upper Bound": ci.iloc[:, 1].values.round(2),
        })
        st.dataframe(fc_table, use_container_width=True)

        m1, m2 = st.columns(2)
        m1.metric("Model MAE (holdout)", f"${mae:,.2f}")
        m2.metric("Model RMSE (holdout)", f"${rmse:,.2f}")
    else:
        st.warning("Not enough data to fit a model for this selection.")

# ---------------------------------------------------------------------------
# PAGE 3 — Anomaly Report
# ---------------------------------------------------------------------------
elif page == "3. Anomaly Report":
    st.title("Anomaly Report")
    st.caption("Two independent detection methods: Isolation Forest (global outliers) and rolling Z-score (local surprises).")

    weekly = get_anomalies(df)

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=weekly["week"], y=weekly["sales"], mode="lines", name="Weekly Sales",
                              line=dict(color="#6b7280")))
    iso_anom = weekly[weekly["iso_anomaly"] == -1]
    fig.add_trace(go.Scatter(x=iso_anom["week"], y=iso_anom["sales"], mode="markers", name="Isolation Forest Anomaly",
                              marker=dict(color="#dc2626", size=12, symbol="x")))
    z_anom = weekly[weekly["z_anomaly"]]
    fig.add_trace(go.Scatter(x=z_anom["week"], y=z_anom["sales"], mode="markers", name="Z-Score Anomaly",
                              marker=dict(color="#9333ea", size=12, symbol="diamond")))
    fig.update_layout(title="Weekly Sales with Detected Anomalies", xaxis_title="Week", yaxis_title="Sales ($)")
    st.plotly_chart(fig, use_container_width=True)

    col1, col2 = st.columns(2)
    with col1:
        st.subheader(f"Isolation Forest Anomalies ({len(iso_anom)})")
        st.dataframe(iso_anom[["week", "sales"]].assign(week=lambda d: d["week"].dt.strftime("%Y-%m-%d")),
                     use_container_width=True)
    with col2:
        st.subheader(f"Z-Score Anomalies ({len(z_anom)})")
        st.dataframe(z_anom[["week", "sales", "zscore"]].assign(week=lambda d: d["week"].dt.strftime("%Y-%m-%d")).round(2),
                     use_container_width=True)

    both = weekly[(weekly["iso_anomaly"] == -1) & (weekly["z_anomaly"])]
    st.info(f"Both methods agree on **{len(both)}** week(s) — the two methods largely disagree because they answer "
            f"different questions: Isolation Forest flags global outliers in raw sales, while Z-score flags local "
            f"surprises relative to the recent trailing trend.")

# ---------------------------------------------------------------------------
# PAGE 4 — Product Demand Segments
# ---------------------------------------------------------------------------
elif page == "4. Product Demand Segments":
    st.title("Product Demand Segments")
    st.caption("K-Means clustering (k=4) on sub-category level total sales, volatility, growth rate, and average order value.")

    feat_df = get_clusters(df)

    palette = {"High Volume, Stable Demand": "#2563eb", "Growing Demand": "#16a34a",
               "Declining Demand": "#dc2626", "Low Volume, High Volatility": "#f97316"}

    fig = px.scatter(feat_df, x="PC1", y="PC2", color="ClusterLabel", text="SubCategory",
                      color_discrete_map=palette, size="TotalSales", size_max=40)
    fig.update_traces(textposition="top center")
    fig.update_layout(title="Sub-Category Clusters (PCA Projection)")
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("Sub-Category → Cluster Mapping")
    display_df = feat_df[["SubCategory", "ClusterLabel", "TotalSales", "GrowthRate", "AvgOrderValue"]].copy()
    display_df["TotalSales"] = display_df["TotalSales"].round(0)
    display_df["GrowthRate"] = (display_df["GrowthRate"] * 100).round(1).astype(str) + "%"
    display_df["AvgOrderValue"] = display_df["AvgOrderValue"].round(2)
    display_df.columns = ["Sub-Category", "Demand Cluster", "Total Sales ($)", "YoY Growth", "Avg Order Value ($)"]
    st.dataframe(display_df.sort_values("Demand Cluster"), use_container_width=True)

    st.subheader("Recommended Stocking Strategy")
    st.markdown("""
    - **High Volume, Stable Demand** — maintain steady safety stock sized to historical demand; standard reorder-point systems work well.
    - **Growing Demand** — increase stock allocation ahead of trend; understocking is costly given high average order values in this cluster.
    - **Declining Demand** — reduce future stock commitments; consider clearance pricing to free up tied-up capital.
    - **Low Volume, High Volatility** — keep lean stock with faster reorder cycles rather than large batch orders.
    """)

st.sidebar.markdown("---")
st.sidebar.caption("Superstore Sales Forecasting & Demand Intelligence System")
