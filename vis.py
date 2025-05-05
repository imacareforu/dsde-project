import streamlit as st
import pandas as pd
import pydeck as pdk
import plotly.express as px
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from datetime import datetime

st.set_page_config(page_title="Bangkok Airbnb Analysis", layout="wide")
st.title('Bangkok Airbnb Listings Analysis')

# Load and prepare data
@st.cache_data
def load_data():
    data = pd.read_csv('bangkok_traffy.csv')
    
    # Clean and prepare data
    data = data.dropna()
    return data

# Load data
data = load_data()

# Sidebar
st.sidebar.header('Options')

#---------------------------------hostogram------------------------------------
# 1. ล้าง format: เอา { } ออก แล้วแยกด้วย comma
data['type_list'] = data['type'].str.strip("{}").str.split(",")

# 2. แปลงเป็น flat list ด้วย explode()
all_types = data.explode('type_list')['type_list'].str.strip()

# 3. นับจำนวนแต่ละ type
type_counts = all_types.value_counts().reset_index()
type_counts.columns = ['type', 'count']

# 4. วาดกราฟด้วย Plotly
fig = px.bar(type_counts, x='type', y='count', title='จำนวนแต่ละประเภทในคอลัมน์ type', text='count')
fig.update_traces(textposition='outside')
fig.update_layout(xaxis_title='ประเภท', yaxis_title='จำนวน')

# 5. แสดงใน Streamlit
st.plotly_chart(fig)

# ---------------------------------- Before/After Gallery ----------------------------------
data['timestamp'] = pd.to_datetime(data['timestamp'], format='mixed', errors='coerce')
data['year'] = data['timestamp'].dt.year
data['month'] = data['timestamp'].dt.month

data = data.dropna(subset=['timestamp'])

# Sidebar filters
selected_year = st.sidebar.selectbox("Select Year", sorted(data['year'].unique(), reverse=True))
selected_month = st.sidebar.selectbox("Select Month", sorted(data[data['year'] == selected_year]['month'].unique()))
selected_district = st.sidebar.selectbox("Select District", sorted(data['district'].dropna().unique()))

filtered_data = data[
    (data['year'] == selected_year) &
    (data['month'] == selected_month) &
    (data['district'] == selected_district)
]

# Split the 'coords' column by comma and convert to float
coords_extracted = filtered_data['coords'].str.split(",", expand=True)

# Assign the split parts to lat and lon
filtered_data['lon'] = pd.to_numeric(coords_extracted[0], errors='coerce')
filtered_data['lat'] = pd.to_numeric(coords_extracted[1], errors='coerce')

filtered_data['photo_after'] = filtered_data['photo_after'].fillna("")

# Define color mapping for states
state_colors = {
    'เสร็จสิ้น': [0, 255, 0, 255],       
    'กำลังดำเนินการ': [255, 255, 0, 255],    
    'รอรับเรื่อง': [255, 0, 0, 255] 
}

# Map the 'state' column to corresponding colors
filtered_data['color'] = filtered_data['state'].map(state_colors)

st.subheader("Issue Reports Map")
col1, col2 = st.columns(2)
with col1:
    st.metric("Filtered Report Count", f"{len(filtered_data):,}")

tooltip = {
    "html": """
    <b>Ticket ID:</b> {ticket_id} <br/>
    <b>State:</b> {state} <br/>
    <b>Type:</b> {type} <br/>
    <b>Comment:</b> {comment} <br/>
    <b>District:</b> {district} <br/>
    <b>Before:</b><br/>
    <img src="{photo}" width="200px"><br/>
    <b>After:</b><br/>
    <img src="{photo_after}" width="200px">
    """,
    "style": {"backgroundColor": "white", "color": "black"}
}

layer = pdk.Layer(
    "ScatterplotLayer",
    data=filtered_data,
    get_position='[lon, lat]',
    get_radius=80,  
    get_color='color', 
    pickable=True
)


view_state = pdk.ViewState(
    latitude=filtered_data['lat'].mean(),
    longitude=filtered_data['lon'].mean(),
    zoom=11,
    pitch=0
)

r = pdk.Deck(
    layers=[layer],
    initial_view_state=view_state,
    tooltip=tooltip
)

st.pydeck_chart(r, use_container_width=True, height=1000)
