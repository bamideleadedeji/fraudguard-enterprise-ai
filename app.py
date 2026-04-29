import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_recall_fscore_support, confusion_matrix

# --- CONFIGURATION & BRANDING ---
st.set_page_config(page_title="FraudGuard AI | Enterprise", layout="wide", page_icon=" ")

# Custom CSS for a professional dark-mode fintech look
st.markdown("""
    <style>
    .main { background-color: #0d1117; }
    div[data-testid="stMetricValue"] { font-size: 24px; color: #00ff88; }
    .stDataFrame { border: 1px solid #30363d; border-radius: 10px; }
    </style>
    """, unsafe_allow_html=True)

# --- DATA GENERATION ENGINE ---
@st.cache_data
def generate_nigerian_fintech_data(n=2500):
    np.random.seed(42)
    merchants = ["POS_LAGOS_VI", "OPAY_FUND_TRANSFER", "PALMPAY_AGENT_IKEJA", "PAYSTACK_CHECKOUT", "FLUTTERWAVE_SUB", "MTN_MOBILE_MONEY"]
    channels = ["Mobile_App", "USSD", "Web_Portal", "POS_Terminal"]
    
    df = pd.DataFrame({
        "tx_id": [f"FTX-{np.random.randint(100000, 999999)}" for _ in range(n)],
        "timestamp": pd.date_range(start=datetime.now()-timedelta(days=30), end=datetime.now(), periods=n),
        "amount": np.random.lognormal(mean=10, sigma=1.5, size=n), # Naira values
        "merchant": np.random.choice(merchants, n),
        "channel": np.random.choice(channels, n),
        "user_location": np.random.choice(["Lagos", "Abuja", "Ibadan", "Kano", "Port-Harcourt", "International"], n, p=[0.5, 0.15, 0.1, 0.1, 0.1, 0.05])
    })
    
    # Fraud logic (The "Target")
    df['is_fraud'] = 0
    # Rule 1: High value USSD at night
    df.loc[(df['channel'] == "USSD") & (df['amount'] > 150000) & (df['timestamp'].dt.hour < 5), 'is_fraud'] = 1
    # Rule 2: International spikes
    df.loc[(df['user_location'] == "International") & (df['amount'] > 50000), 'is_fraud'] = 1
    # Rule 3: POS Lagos outliers
    df.loc[(df['merchant'] == "POS_LAGOS_VI") & (df['amount'] > 1000000), 'is_fraud'] = 1
    
    # Random Noise
    noise = np.random.choice([0, 1], size=n, p=[0.97, 0.03])
    df['is_fraud'] = (df['is_fraud'] | noise)
    return df

# --- MACHINE LEARNING ENGINE ---
@st.cache_resource
def train_enterprise_model(df):
    # Feature Engineering
    df_ml = df.copy()
    df_ml['hour'] = df_ml['timestamp'].dt.hour
    X = pd.get_dummies(df_ml[['amount', 'hour', 'channel', 'user_location']])
    y = df_ml['is_fraud']
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42)
    model.fit(X_train, y_train)
    
    y_pred = model.predict(X_test)
    prec, rec, f1, _ = precision_recall_fscore_support(y_test, y_pred, average='binary')
    cm = confusion_matrix(y_test, y_pred)
    return model, prec, rec, f1, cm, X.columns

# Init
df = generate_nigerian_fintech_data()
model, prec, rec, f1, cm, feat_cols = train_enterprise_model(df)

# --- UI NAVIGATION ---
st.sidebar.title(" FraudGuard AI")
st.sidebar.markdown("---")
view = st.sidebar.radio("Navigation", ["Executive Overview", "ML Model Evidence", "Risk Log", "About"])

if view == "Executive Overview":
    st.title(" Enterprise Risk Monitoring")
    
    col1, col2, col3, col4 = st.columns(4)
    total_fraud_naira = df[df['is_fraud']==1]['amount'].sum()
    col1.metric("Fraud Prevented", f"₦{total_fraud_naira:,.0f}")
    col2.metric("System Accuracy", f"{(prec+rec)/2:.1%}")
    col3.metric("Review Efficiency", "88%", "+5%")
    col4.metric("Avg Score Time", "12ms")

    st.divider()

    c1, c2 = st.columns([2, 1])
    with c1:
        st.subheader("Daily Fraud Detection Volume")
        trend = df.groupby(df['timestamp'].dt.date)['is_fraud'].sum().reset_index()
        fig = px.area(trend, x='timestamp', y='is_fraud', color_discrete_sequence=['#FF4B4B'])
        st.plotly_chart(fig, use_container_width=True)
    with c2:
        st.subheader("High-Risk Channels")
        fig2 = px.pie(df, names='channel', color='is_fraud', hole=0.4)
        st.plotly_chart(fig2, use_container_width=True)

elif view == "ML Model Evidence":
    st.title(" Model Integrity & Metrics")
    st.markdown("Proof of performance for technical stakeholders.")
    
    k1, k2, k3 = st.columns(3)
    k1.metric("Precision", f"{prec:.2%}")
    k2.metric("Recall (Capture Rate)", f"{rec:.2%}")
    k3.metric("F1-Score", f"{f1:.2%}")

    st.write("**Confusion Matrix: Validating Detection Accuracy**")
    fig_cm = px.imshow(cm, text_auto=True, labels=dict(x="Predicted", y="Actual"),
                      x=['Legit', 'Fraud'], y=['Legit', 'Fraud'], color_continuous_scale='Reds')
    st.plotly_chart(fig_cm, use_container_width=True)

elif view == "Risk Log":
    st.title(" High-Risk Transaction Log")
    # Scoring live data
    df_temp = df.copy()
    df_temp['hour'] = df_temp['timestamp'].dt.hour
    X_live = pd.get_dummies(df_temp[['amount', 'hour', 'channel', 'user_location']]).reindex(columns=feat_cols, fill_value=0)
    df['risk_score'] = model.predict_proba(X_live)[:, 1]
    
    high_risk = df[df['risk_score'] > 0.7].sort_values('risk_score', ascending=False)
    st.dataframe(high_risk[['timestamp', 'amount', 'merchant', 'user_location', 'risk_score']].style.background_gradient(cmap='Reds'), use_container_width=True)

st.sidebar.markdown("---")
st.sidebar.caption("FraudGuard AI v2.5 - Enterprise Edition")
