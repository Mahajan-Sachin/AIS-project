import streamlit as st
import pandas as pd
from ais_model import AISIntrusionDetector

st.set_page_config(page_title="AIS Intrusion Detection", layout="centered")
st.title("🛡️ AIS-based Intrusion Detection System")

st.write("""
Upload your network traffic dataset (CSV format) to detect potential intrusions
using an Artificial Immune System (AIS)-inspired model.
""")

# Upload CSV file
uploaded_file = st.file_uploader("Choose a CSV file", type="csv")

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    st.write("### 🔍 Preview of Uploaded Data")
    st.dataframe(df.head())

    detector = AISIntrusionDetector()
    detector.fit(df)
    results = detector.predict(df)

    st.write("### 🧠 Detection Results")
    st.dataframe(results[['anomaly', 'risk_score', 'alert']])

    intrusions = results[results['alert'] == 'Intrusion']
    st.write(f"### 🚨 Total Intrusions Detected: {len(intrusions)}")

    st.download_button("📥 Download Results as CSV", data=results.to_csv(index=False),
                       file_name="detection_results.csv", mime="text/csv")
