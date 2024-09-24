import streamlit as st
import pandas as pd
import numpy as np
import altair as alt
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
import time

# Simulated function to load data - replace with actual data source
def load_data():
    # Sample data: replace this with your real-time logs or network traffic data
    data = pd.DataFrame({
        'timestamp': pd.date_range(start='2024-09-23', periods=100, freq='T'),
        'ip_address': np.random.choice(['192.168.1.1', '192.168.1.2', '192.168.1.3'], size=100),
        'packet_size': np.random.randint(50, 1500, size=100),
        'user_activity_score': np.random.uniform(0, 1, size=100),
        'response_time': np.random.uniform(0.1, 5, size=100)
    })
    return data

# Preprocess the data for the ML model
def preprocess_data(data):
    # Extract features from the data
    features = data[['packet_size', 'user_activity_score', 'response_time']]
    
    # Standardize the features
    scaler = StandardScaler()
    scaled_features = scaler.fit_transform(features)
    
    return scaled_features

# Train or load anomaly detection model
def train_anomaly_detection_model(data):
    model = IsolationForest(n_estimators=100, contamination=0.05, random_state=42)
    model.fit(data)
    return model

# Detect anomalies in new data
def detect_anomalies(model, data):
    predictions = model.predict(data)
    # IsolationForest labels normal data as 1 and anomalies as -1
    anomaly_scores = model.decision_function(data)
    return predictions, anomaly_scores

# Block IP (for simulation purposes - in practice, integrate with firewall APIs)
def block_ip(ip_address):
    st.write(f"Blocking IP address: {ip_address}")
    # Implement actual firewall blocking logic here

# Streamlit app code
st.title("AI-driven Cybersecurity Monitoring System")
st.markdown("""
### Real-time Detection of Anomalies in Network Traffic or System Logs
This application monitors incoming network data, identifies anomalies, and automates response actions like blocking suspicious IP addresses. 
""")

# Sidebar - Manual Threat Response
st.sidebar.title("Manual Threat Response")
manual_block_ip = st.sidebar.text_input("Enter IP to Block:", "")
if st.sidebar.button("Block IP"):
    block_ip(manual_block_ip)

# Load data
st.subheader("Incoming Data")
data = load_data()
st.write(data.tail())

# Preprocess the data for the model
scaled_data = preprocess_data(data)

# Train the model
model = train_anomaly_detection_model(scaled_data)

# Perform anomaly detection on the data
predictions, anomaly_scores = detect_anomalies(model, scaled_data)

# Add results back to the data frame
data['anomaly'] = predictions
data['anomaly_score'] = anomaly_scores

# Filter for anomalies (-1 means anomaly, 1 means normal)
anomalies = data[data['anomaly'] == -1]
normal_data = data[data['anomaly'] == 1]

# Real-time visualization of anomalies
st.subheader("Detected Anomalies")
if len(anomalies) > 0:
    st.write(anomalies[['timestamp', 'ip_address', 'packet_size', 'anomaly_score']])
else:
    st.write("No anomalies detected.")

# Plot anomalies using Altair
st.subheader("Anomaly Score Over Time")
chart = alt.Chart(data).mark_line().encode(
    x='timestamp:T',
    y='anomaly_score:Q',
    color=alt.condition(
        alt.datum.anomaly == -1, alt.value('red'), alt.value('blue')  # Color anomalies in red
    )
).interactive()

st.altair_chart(chart, use_container_width=True)

# Automate responses for detected anomalies
st.subheader("Automated Responses")
for ip in anomalies['ip_address'].unique():
    st.write(f"IP {ip} detected as suspicious. Automatically blocking...")
    block_ip(ip)

# Refresh the data every minute (simulate real-time behavior)
refresh_rate = st.slider("Refresh rate (seconds):", 1, 60, 10)
st.write(f"Data will refresh every {refresh_rate} seconds.")
time.sleep(refresh_rate)

# Optionally, you can trigger real-time data refresh and model re-evaluation here in a loop
