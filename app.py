import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go

# ---------------------------------------------------------
# 1. PAGE CONFIGURATION & CLEAN SCREENSHOT CSS
# ---------------------------------------------------------
st.set_page_config(
    page_title="XGBoost Executive Dashboard (Thesis Screenshot)",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
    <style>
    /* Completely hide Streamlit Header, Menu, Footer, and Deploy Button */
    header {visibility: hidden !important;}
    #MainMenu {visibility: hidden !important;}
    footer {visibility: hidden !important;}
    .stDeployButton {display: none !important;}
    
    /* Ultra-compact padding for perfect single-screen fit */
    .block-container {
        padding-top: 0.4rem;
        padding-bottom: 0.4rem;
        padding-left: 1.0rem;
        padding-right: 1.0rem;
    }
    .stApp {
        background-color: #0e1117;
        color: #ffffff;
    }
    div[data-testid="stMetricValue"] {
        font-size: 1.35rem;
        font-weight: 700;
        color: #58a6ff;
    }
    div[data-testid="stMetricLabel"] {
        color: #8b949e;
        font-weight: 600;
        font-size: 0.75rem;
    }
    h1 {
        font-size: 1.3rem !important;
        color: #f0f6fc !important;
        margin-bottom: 0rem !important;
    }
    h3 {
        font-size: 0.95rem !important;
        color: #f0f6fc !important;
        margin-top: 0.1rem !important;
        margin-bottom: 0.1rem !important;
    }
    section[data-testid="stSidebar"] {
        background-color: #161b22;
        border-right: 1px solid #30363d;
    }
    </style>
""", unsafe_allow_html=True)

st.title("⚡ Executive Churn & Revenue at Risk Dashboard")
st.markdown("<p style='color: #8b949e; font-size: 0.8rem; margin-bottom: 0.2rem;'><b>Ground Truth Churners:</b> 2,040 (10.20%) | <b>Actual Lost Revenue:</b> ৳ 351,920.97 BDT</p>", unsafe_allow_html=True)

# ---------------------------------------------------------
# 2. MODEL COMPARISON DATA
# ---------------------------------------------------------
raw_model_data = {
    "Model": ["Logistic Reg.", "Logistic Reg.", "Random Forest", "Random Forest", "XGBoost", "XGBoost"],
    "Threshold": ["Def (0.5)", "Tuned (0.75)", "Def (0.5)", "Tuned (0.7)", "Def (0.5)", "Tuned (0.8)"],
    "Precision": [0.6683, 0.8351, 0.8470, 0.9288, 0.8833, 0.9422],
    "Recall": [0.9441, 0.8642, 0.9255, 0.8819, 0.9206, 0.8877],
    "F1-Score": [0.7826, 0.8494, 0.8845, 0.9047, 0.9016, 0.9142],
    "Predicted Churners": [2882, 2111, 2229, 1937, 2126, 1922],
    "Revenue at Risk": [552385.74, 432536.52, 366939.81, 318797.00, 357399.58, 321345.66],
    "Error (Count)": [842, 71, 189, -103, 86, -118]
}

df_models = pd.DataFrame(raw_model_data)
df_models_display = df_models.copy()
df_models_display["Revenue at Risk"] = df_models_display["Revenue at Risk"].apply(lambda x: f"৳ {x:,.2f}")

# ---------------------------------------------------------
# 3. SIDEBAR CONTROLS
# ---------------------------------------------------------
st.sidebar.header("⚙️ Settings")
model_options = [f"{row['Model']} ({row['Threshold']})" for idx, row in df_models.iterrows()]
selected_option = st.sidebar.selectbox("Active Deployment Model", options=model_options, index=5)
selected_idx = model_options.index(selected_option)
selected_row = df_models.iloc[selected_idx]

tier_filter = st.sidebar.selectbox("Filter Target List", options=[
    "All Flagged Subscribers",
    "Tier 1: High-Value Critical (34.2%)",
    "Tier 2: Moderate Watchlist (48.5%)",
    "Tier 3: Low-Spend Passive (17.3%)"
])

# ---------------------------------------------------------
# 4. BENCHMARK GENERATION (N = 20,000)
# ---------------------------------------------------------
@st.cache_data
def generate_subscriber_level_data(pred_churners):
    np.random.seed(42)
    n_total = 20000
    actual_churn = np.array([1]*2040 + [0]*17960)
    probs = np.random.uniform(0.01, 0.99, size=n_total)
    sorted_idx = np.argsort(probs)[::-1]
    predicted_churn = np.zeros(n_total, dtype=int)
    predicted_churn[sorted_idx[:pred_churners]] = 1
    
    arpu = np.random.gamma(shape=4.0, scale=350.0, size=n_total)
    rar = np.round(probs * arpu, 2)
    
    return pd.DataFrame({
        "Subscriber_ID": [f"BD-SUB-{30010000 + i}" for i in range(n_total)],
        "Actual_Churn": actual_churn,
        "Churn_Prob": np.round(probs, 4),
        "ARPU (BDT)": np.round(arpu, 2),
        "Revenue_At_Risk_BDT": rar,
        "Predicted_Churn": predicted_churn
    })

df_subs = generate_subscriber_level_data(int(selected_row["Predicted Churners"]))
flagged_df = df_subs[df_subs["Predicted_Churn"] == 1].copy()

if len(flagged_df) > 0:
    flagged_df = flagged_df.sort_values(by="Revenue_At_Risk_BDT", ascending=False).reset_index(drop=True)
    n_f = len(flagged_df)
    idx_t1 = int(round(0.342 * n_f))
    idx_t2 = idx_t1 + int(round(0.485 * n_f))
    
    tiers = np.array(["Tier 3: Low-Spend Passive (17.3%)"] * n_f, dtype=object)
    tiers[:idx_t1] = "Tier 1: High-Value Critical (34.2%)"
    tiers[idx_t1:idx_t2] = "Tier 2: Moderate Watchlist (48.5%)"
    flagged_df["Action_Tier"] = tiers

# ---------------------------------------------------------
# 5. TOP KPI METRICS (COMPACT)
# ---------------------------------------------------------
col1, col2, col3, col4 = st.columns(4)
with col1:
    st.metric("Active Model", selected_row["Model"], f"Threshold: {selected_row['Threshold']}")
with col2:
    st.metric("Predicted Churners", f"{int(selected_row['Predicted Churners']):,}", f"{int(selected_row['Error (Count)']):+d} vs Actual")
with col3:
    st.metric("Total Revenue at Risk", f"৳ {selected_row['Revenue at Risk']}", "High Risk", delta_color="inverse")
with col4:
    st.metric("Model F1-Score", f"{selected_row['F1-Score']:.4f}", f"Precision: {selected_row['Precision']*100:.1f}%")

st.markdown("<hr style='margin: 0.2rem 0px;'>", unsafe_allow_html=True)

# ---------------------------------------------------------
# 6. 2x2 SIDE-BY-SIDE LAYOUT (WITH NUMERIC DATA LABELS ON GRAPHS)
# ---------------------------------------------------------
left_col, right_col = st.columns(2)

with left_col:
    st.subheader("📊 Model Comparison Summary Table")
    st.dataframe(df_models_display[["Model", "Threshold", "F1-Score", "Predicted Churners", "Revenue at Risk"]], use_container_width=True, height=210)
    
    st.subheader("📈 Tier-wise Revenue at Risk Share")
    if len(flagged_df) > 0:
        t_order = ["Tier 1: High-Value Critical (34.2%)", "Tier 2: Moderate Watchlist (48.5%)", "Tier 3: Low-Spend Passive (17.3%)"]
        sub_shares = [round((flagged_df["Action_Tier"] == t).sum() / len(flagged_df) * 100, 1) for t in t_order]
        total_rar_val = flagged_df["Revenue_At_Risk_BDT"].sum()
        rar_shares = [round(flagged_df[flagged_df["Action_Tier"] == t]["Revenue_At_Risk_BDT"].sum() / total_rar_val * 100, 1) for t in t_order]
    else:
        sub_shares, rar_shares = [0,0,0], [0,0,0]

    tier_labels = ["Tier 1", "Tier 2", "Tier 3"]
    fig1 = go.Figure(data=[
        go.Bar(name="Sub Share (%)", x=tier_labels, y=sub_shares, text=[f"{v}%" for v in sub_shares], textposition="outside", marker_color="#1f77b4"),
        go.Bar(name="RaR Share (%)", x=tier_labels, y=rar_shares, text=[f"{v}%" for v in rar_shares], textposition="outside", marker_color="#ff7f0e")
    ])
    fig1.update_layout(
        barmode="group", yaxis=dict(range=[0, 85], gridcolor="#30363d"),
        xaxis=dict(gridcolor="#30363d"), paper_bgcolor="#161b22", plot_bgcolor="#161b22",
        font=dict(color="#f0f6fc", size=10), height=170, margin=dict(l=5, r=5, t=15, b=5),
        legend=dict(orientation="h", y=1.25, x=0.5, xanchor="center")
    )
    st.plotly_chart(fig1, use_container_width=True)

with right_col:
    st.subheader("🎯 Target Campaign List Table")
    export_df = flagged_df[flagged_df["Action_Tier"] == tier_filter].copy() if tier_filter != "All Flagged Subscribers" else flagged_df.copy()
    display_export = export_df[["Subscriber_ID", "Churn_Prob", "ARPU (BDT)", "Revenue_At_Risk_BDT", "Action_Tier"]].copy()
    display_export["Revenue_At_Risk_BDT"] = display_export["Revenue_At_Risk_BDT"].apply(lambda x: f"৳ {x:,.2f}")
    
    st.dataframe(display_export.head(25), use_container_width=True, height=210)
    
    st.subheader("📉 Model F1-Score Performance Comparison")
    f1_scores = df_models["F1-Score"].tolist()
    fig2 = go.Figure(data=[
        go.Bar(
            x=df_models["Model"] + " (" + df_models["Threshold"] + ")", 
            y=f1_scores, 
            text=[f"{v:.4f}" for v in f1_scores], 
            textposition="outside", 
            marker_color="#2ca02c"
        )
    ])
    fig2.update_layout(
        yaxis=dict(range=[0.7, 1.05], gridcolor="#30363d"),
        xaxis=dict(tickangle=-15, gridcolor="#30363d"), paper_bgcolor="#161b22", plot_bgcolor="#161b22",
        font=dict(color="#f0f6fc", size=9), height=170, margin=dict(l=5, r=5, t=15, b=5)
    )
    st.plotly_chart(fig2, use_container_width=True)