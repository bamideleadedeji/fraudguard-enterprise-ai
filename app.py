@st.cache_data(ttl=10)
def ingest_cloud_audit_matrix(file_path, _key_file_anchor):
    """Reads the audit ledger locally from the repository, bypassing cloud architecture completely."""
    try:
        local_file = "client_dejifolakemi_enterprises_extracted_ledger.csv"
        if os.path.exists(local_file):
            df_raw = pd.read_csv(local_file)
            df_raw.columns = [str(col).strip().lower() for col in df_raw.columns]
            
            if 'timestamp' in df_raw.columns:
                df_raw['timestamp'] = pd.to_datetime(df_raw['timestamp'], errors='coerce')
            if 'risk_score' in df_raw.columns:
                df_raw['risk_score'] = pd.to_numeric(df_raw['risk_score'], errors='coerce').fillna(0.1500)
                df_raw['is_fraud'] = np.where(df_raw['risk_score'] > 0.85, 1, 0)
            if 'amount' in df_raw.columns:
                df_raw['amount'] = pd.to_numeric(df_raw['amount'], errors='coerce').fillna(0.0)
            return df_raw
        else:
            return pd.DataFrame()
    except Exception as e:
        st.error(f"Data Ingestion Failure: {e}")
        return pd.DataFrame()
