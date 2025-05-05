import streamlit as st
import pandas as pd
import pydeck as pdk
import geopandas as gpd
import plotly.express as px
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from pymongo import MongoClient
import numpy as np
import json
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
#---------------------------------other function-------------------------------
def haversine_np(lon1, lat1, lon2, lat2):
    R = 6371  # Earth radius in km
    lon1, lat1, lon2, lat2 = map(np.radians, [lon1, lat1, lon2, lat2])
    dlon = lon2 - lon1
    dlat = lat2 - lat1

    a = np.sin(dlat / 2.0)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2.0)**2
    return R * 2 * np.arcsin(np.sqrt(a))


#---------------------------------histogram------------------------------------
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
all_types = data.explode('type_list')['type_list'].str.strip().dropna().unique()
all_types = sorted(all_types)

data['timestamp'] = pd.to_datetime(data['timestamp'], format='mixed', errors='coerce')
data['year'] = data['timestamp'].dt.year
data['month'] = data['timestamp'].dt.month

data = data.dropna(subset=['timestamp'])

with st.sidebar.form("filter_form"):
    selected_year = st.selectbox("Select Year", sorted(data['year'].unique(), reverse=True))
    all_months = list(range(1, 13))
    available_months = sorted(data[data['year'] == selected_year]['month'].unique())
    default_month = next((m for m in available_months if m in all_months), 1)
    selected_month = st.selectbox("Select Month", all_months, index=all_months.index(default_month))
    selected_districts = st.multiselect("Select District(s)",options=sorted(data['district'].dropna().unique()),default=[])
    selected_types = st.multiselect("Select Type(s)", all_types)

    # Submit button
    submitted = st.form_submit_button("Apply Filters")

if submitted:
    if selected_month not in available_months:
        st.warning(f"Month {selected_month} has no data in year {selected_year}. Please select another month.")
        st.stop()

    filtered_data = data[
        (data['year'] == selected_year) &
        (data['month'] == selected_month)
    ]

    if selected_districts:
        filtered_data = filtered_data[filtered_data['district'].isin(selected_districts)]

    if selected_types:
        # Explode the 'type_list' column to allow multi-type filtering
        filtered_data = filtered_data.explode('type_list')
        filtered_data['type_list'] = filtered_data['type_list'].str.strip()
        filtered_data = filtered_data[filtered_data['type_list'].isin(selected_types)]

    # Proceed with map, charts, etc.
else:
    st.stop()  # Prevent rest of app from running before form submission

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

# For issue reports
filtered_data['tooltip_html'] = (
    "<b>Ticket ID:</b> " + filtered_data['ticket_id'].astype(str) + "<br/>"
    "<b>State:</b> " + filtered_data['state'] + "<br/>"
    "<b>Type:</b> " + filtered_data['type'].str.strip("{}") + "<br/>"
    "<b>Comment:</b> " + filtered_data['comment'] + "<br/>"
    "<b>District:</b> " + filtered_data['district'] + "<br/>"
    "<b>Before:</b><br/><img src='" + filtered_data['photo'] + "' width='200px'><br/>"
    "<b>After:</b><br/><img src='" + filtered_data['photo_after'] + "' width='200px'>"
)

st.subheader("Issue Reports Map")
col1, col2 = st.columns(2)
with col1:
    st.metric("Filtered Report Count", f"{len(filtered_data):,}")

layer_reports = pdk.Layer(
    "ScatterplotLayer",
    data=filtered_data,
    get_position='[lon, lat]',
    get_radius=150,
    get_color='color',
    pickable=True
)

#---------------------------external data----------------------

# Load geojson
with open('school_in_bangkok.geojson', encoding='utf-8') as f:
    geojson_data = json.load(f)

# Extract feature data
features = geojson_data['features']

# Extract data into list of dicts
records = []
for feature in features:
    prop = feature['properties']
    coords = feature['geometry']['coordinates']
    record = {
        'name': prop.get('name'),
        'lon': coords[0],
        'lat': coords[1]
    }
    records.append(record)

# Convert to DataFrame
df_geo = pd.DataFrame(records)

df_geo['tooltip_html'] = (
    "<b>Name:</b> " + df_geo['name']
)
# Filter schools within 2 km of any issue report
report_lats = filtered_data['lat'].values
report_lons = filtered_data['lon'].values

school_distances = []
for _, row in df_geo.iterrows():
    dists = haversine_np(report_lons, report_lats, row['lon'], row['lat'])
    min_dist = dists.min()
    school_distances.append(min_dist)

df_geo['min_dist_km'] = school_distances
df_geo = df_geo[df_geo['min_dist_km'] <= 2]

layer_schools = pdk.Layer(
    "ScatterplotLayer",
    data=df_geo,
    get_position='[lon, lat]',
    get_radius=100,
    get_color=[255, 255, 255, 255],
    pickable=True
)

#---------------------------combine data--------------------
tooltip = {
    "html": "{tooltip_html}",
    "style": {"backgroundColor": "white", "color": "black"}
}

# Use either view_state, depending on what makes more sense — here we use issue reports
combined_view_state = pdk.ViewState(
    latitude=filtered_data['lat'].mean(),
    longitude=filtered_data['lon'].mean(),
    zoom=11,
    pitch=0
)

# Combine tooltips using a shared tooltip or leave as-is (each layer can have its own)
r = pdk.Deck(
    layers=[layer_reports,layer_schools],
    initial_view_state=combined_view_state,
    tooltip=tooltip
)

# Show in Streamlit
st.pydeck_chart(r, use_container_width=True, height=1000)
