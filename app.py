import streamlit as st
from streamlit_folium import folium_static
import folium
import pandas as pd
import pydeck as pdk
import geopandas as gpd
import plotly.express as px
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from pymongo import MongoClient
import numpy as np
import re
import requests
import json
from folium import Choropleth, GeoJson, GeoJsonTooltip
from datetime import datetime, timedelta
import streamlit.components.v1 as components

st.set_page_config(page_title="traffy fondu dataset analysis", layout="wide")
st.title('Traffy Fondu Dataset Analysis')

# Load and prepare data
@st.cache_data
def load_data():
    data = pd.read_csv('bangkok_traffy.csv')
    
    # Clean and prepare data
    data = data.dropna()
    return data

# Load data
data = load_data()

tab1, tab2 = st.tabs(["📊 Overall", "📎 Statistic"])

with tab1:
    # ----------------------------- สรุปจำนวนปัญหา -----------------------------
    # เตรียมข้อมูล
    total_issues = len(data)
    completed_issues = len(data[data['state'] == 'เสร็จสิ้น'])
    incomplete_issues = total_issues - completed_issues

    col1, col2, col3 = st.columns(3)
    col1.metric("📌 Total reported issues", f"{total_issues:,}")
    col2.metric("✅ Issues Resolved", f"{completed_issues:,}")
    col3.metric("⏳ Unresolved Issues", f"{incomplete_issues:,}")


#---------------------------------other function-------------------------------
def haversine_np(lon1, lat1, lon2, lat2):
    R = 6371  # Earth radius in km
    lon1, lat1, lon2, lat2 = map(np.radians, [lon1, lat1, lon2, lat2])
    dlon = lon2 - lon1
    dlat = lat2 - lat1

    a = np.sin(dlat / 2.0)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2.0)**2
    return R * 2 * np.arcsin(np.sqrt(a))

with tab2:
    #---------------------------------histogram------------------------------------
    st.subheader("Distribution of Issue Types")
    # 1. ล้าง format: เอา { } ออก แล้วแยกด้วย comma
    data['type_list'] = data['type'].str.strip("{}").str.split(",")

    # 2. แปลงเป็น flat list ด้วย explode()
    all_types = data.explode('type_list')['type_list'].str.strip()
    all_types = all_types.replace('', 'ไม่มีประเภท')

    # 3. นับจำนวนแต่ละ type
    type_counts = all_types.value_counts().reset_index()
    type_counts.columns = ['type', 'count']

    # 4. วาดกราฟด้วย Plotly
    fig = px.bar(type_counts, x='type', y='count', text='count')
    fig.update_traces(textposition='outside')
    fig.update_layout(xaxis_title='type', yaxis_title='number of issues')

    # 5. แสดงใน Streamlit
    st.plotly_chart(fig)

    # ---------------------------- Histogram: จำนวนปัญหาในแต่ละเขต -------------------
    st.subheader("Distribution of Issues Across Districts")

    # นับจำนวนในแต่ละเขต (กรอง NaN ออก)
    district_counts = data['district'].dropna().value_counts().reset_index()
    district_counts.columns = ['district', 'count']

    # วาดกราฟแท่งด้วย Plotly
    fig_district = px.bar(
        district_counts,
        x='district',
        y='count',
        text='count'
    )
    fig_district.update_traces(textposition='outside')
    fig_district.update_layout(
        xaxis_title='district',
        yaxis_title='number of issues',
        xaxis_tickangle=-45
    )

    # แสดงกราฟใน Streamlit
    st.plotly_chart(fig_district)

with tab1:
# ----------------------------- pie chart -----------------------------

    # แยก Top N กับที่เหลือ
    top_types = type_counts.head(5)
    others = pd.DataFrame([{
        'type': 'อื่นๆ',
        'count': type_counts['count'][5:].sum()
    }])

    # รวมกลับ
    type_counts_pie = pd.concat([top_types, others], ignore_index=True)

    # วาด Pie Chart
    st.subheader("Proportion of Issue Types")
    fig_type_pie = px.pie(
        type_counts_pie,
        names='type',
        values='count'
    )
    # เพิ่มชื่อบนกราฟ
    fig_type_pie.update_traces(
        textinfo='label+percent',  # แสดงชื่อ + เปอร์เซ็นต์ + จำนวน
        textposition='inside'
    )
    st.plotly_chart(fig_type_pie)

with tab2:
    # ----------------------------- หน่วยงานที่ได้คะแนน star เฉลี่ยสูงสุด -----------------------------
    st.subheader("Top-Rated Organizations")

    # เตรียมข้อมูล
    org_star_df = data[['organization', 'star']].dropna()
    org_star_df['main_org'] = org_star_df['organization'].apply(lambda x: x.split(',')[0].strip())
    org_summary = org_star_df.groupby('main_org')['star'].agg(['count', 'mean']).reset_index()
    org_summary = org_summary[org_summary['count'] >= 5]  # กรองเฉพาะหน่วยงานที่มีรีวิวมากพอ
    org_summary = org_summary.sort_values(by='mean', ascending=False).head(5)

    # วาดกราฟ
    fig_star_org = px.bar(
        org_summary,
        x='mean',
        y='main_org',
        orientation='h',
        text='mean',
        title='🏆Top 5 Organizations (at least 5 reviews)',
        labels={'mean': 'average rating', 'main_org': 'organization'}
    )
    fig_star_org.update_layout(yaxis=dict(autorange="reversed"))
    fig_star_org.update_traces(texttemplate='%{text:.2f}', textposition='outside')

    # แสดงใน Streamlit
    st.plotly_chart(fig_star_org, use_container_width=False)


# ---------------------------------- Before/After Gallery ----------------------------------
all_types = data.explode('type_list')['type_list'].str.strip().replace('', 'ไม่มีประเภท').dropna().unique()
all_types = sorted(all_types)

data['timestamp'] = pd.to_datetime(data['timestamp'], format='mixed', errors='coerce')
data['year'] = data['timestamp'].dt.year
data['month'] = data['timestamp'].dt.month

data = data.dropna(subset=['timestamp'])

# ----------------------------------- Sidebar ---------------------------------
now = datetime.now()

# Default values
default_year = now.year
default_month = now.month - 1 if now.month > 1 else 12
all_months = list(range(1, 13))
all_districts = sorted(data['district'].dropna().unique())
district_options = ["ทุกเขต (ทั้งกรุงเทพฯ)"] + all_districts
all_types_raw = data.explode('type_list')['type_list'].str.strip().replace('', 'ไม่มีประเภท')
all_types = sorted(all_types_raw.dropna().unique().tolist())
type_options = ["ทุกประเภท"] + all_types

# Sidebar filters (no form)
st.sidebar.header('Options')

selected_year = st.sidebar.selectbox("เลือกปี", sorted(data['year'].unique(), reverse=True), index=0)
default_month_index = all_months.index(default_month)
selected_month = st.sidebar.selectbox("เลือกเดือน", all_months, index=default_month_index)
selected_districts = st.sidebar.multiselect("เลือกเขต", district_options, default=["ทุกเขต (ทั้งกรุงเทพฯ)"])
selected_types = st.sidebar.multiselect("เลือกประเภท", type_options, default=["ทุกประเภท"])

show_schools = st.sidebar.checkbox("แสดงโรงเรียนใกล้จุดปัญหา", value=False)
# ถ้าเลือกให้แสดง → แสดง slider เพิ่ม
radius_km = None  # default ถ้าไม่เปิด
if show_schools:
    radius_km = st.sidebar.slider("รัศมี (กิโลเมตร)", min_value=0.0, max_value=5.0, value=2.0, step=0.5)

# เริ่มต้นกรองตามปีและเดือนที่เลือก
filtered_data = data[
    (data['year'] == selected_year) &
    (data['month'] == selected_month)
]

if "ทุกเขต (ทั้งกรุงเทพฯ)" not in selected_districts:
    filtered_data = filtered_data[filtered_data['district'].isin(selected_districts)]

if "ทุกประเภท" not in selected_types:
    filtered_data = filtered_data.explode('type_list')
    filtered_data['type_list'] = filtered_data['type_list'].str.strip().replace('', 'ไม่มีประเภท')
    filtered_data = filtered_data[filtered_data['type_list'].isin(selected_types)]

# ตัดแถวที่ coords ว่าง หรือไม่มี comma ออก
filtered_data = filtered_data[
    filtered_data['coords'].notna() &
    filtered_data['coords'].str.contains(",", na=False)
]
# แยกเป็น lon, lat อย่างปลอดภัย
coords_extracted = filtered_data['coords'].str.split(",", expand=True)

# เงื่อนไขสำคัญ: ต้องมีอย่างน้อย 2 คอลัมน์ถึงจะทำต่อได้
if coords_extracted.shape[1] >= 2:
    filtered_data['lon'] = pd.to_numeric(coords_extracted.iloc[:, 0], errors='coerce')
    filtered_data['lat'] = pd.to_numeric(coords_extracted.iloc[:, 1], errors='coerce')

    # ตัดค่าที่แปลงพิกัดไม่ได้
    filtered_data = filtered_data.dropna(subset=['lon', 'lat'])
else:
    filtered_data['lon'] = None
    filtered_data['lat'] = None

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

with tab1:

    st.subheader("Issue Reports Map")
    col1, col2 = st.columns(2)
    with col1:
        st.metric("Filtered Report Count", f"{len(filtered_data):,}")

    layer_reports = pdk.Layer(
        "ScatterplotLayer",
        data=filtered_data,
        get_position='[lon, lat]',
        get_radius=75,
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
        if len(report_lats) > 0 and len(report_lons) > 0:
            dists = haversine_np(report_lons, report_lats, row['lon'], row['lat'])
            min_dist = dists.min()
        else:
            min_dist = None
        school_distances.append(min_dist)

    df_geo['min_dist_km'] = school_distances
    df_geo = df_geo.dropna(subset=['min_dist_km'])
    if radius_km is not None:
        df_geo = df_geo[df_geo['min_dist_km'] <= radius_km]

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
        zoom=12,
        pitch=0
    )

    # Combine tooltips using a shared tooltip or leave as-is (each layer can have its own)
    layers_to_show = [layer_reports]
    if show_schools:
        layers_to_show.append(layer_schools)

    r = pdk.Deck(
        layers=layers_to_show,
        initial_view_state=combined_view_state,
        map_style='mapbox://styles/mapbox/dark-v9',
        tooltip=tooltip,
        width=1600,
        height=1000,
    )
    # Show in Streamlit
    st.pydeck_chart(r, use_container_width=True)

    #--------------------------------- DE part ------------------------------------
    # โหลดข้อมูลร้องเรียน
    with open("all_data.json", "r", encoding="utf-8") as f:
        data = json.load(f)

    df = pd.DataFrame(data)

    # ฟังก์ชันแปลงเวลา title_state
    def parse_duration(text):
        if not isinstance(text, str): return None
        if not text.startswith("เสร็จสิ้นใน"): return None
        day_match = re.search(r'(\d+)\s*วัน', text)
        hr_match = re.search(r'(\d+):(\d+)\s*ชม\.|(\d+)\s*ชม\.', text)

        total_hours = 0
        if day_match:
            total_hours += int(day_match.group(1)) * 24
        if hr_match:
            if hr_match.group(1) and hr_match.group(2):
                total_hours += int(hr_match.group(1)) + int(hr_match.group(2)) / 60
            elif hr_match.group(3):
                total_hours += int(hr_match.group(3))

        return total_hours if total_hours > 0 else None

    # ดึงชื่อเขต
    def extract_district(text):
        if isinstance(text, str):
            match = re.search(r"เขต(\S+)", text)
            if match:
                return match.group(1)
        return None

    df["duration_hr"] = df["title_state"].apply(parse_duration)
    df["district"] = df["location_thailand"].apply(extract_district)
    df_clean = df.dropna(subset=["duration_hr", "district"])

    # สร้างตาราง avg ต่อเขต
    df_avg = df_clean.groupby("district")["duration_hr"].mean().reset_index()
    df_avg.columns = ["district", "avg_duration_hr"]
    df_avg["dname"] = "เขต" + df_avg["district"]

    # โหลด GeoJSON เขต 
    geojson_url = "https://raw.githubusercontent.com/pcrete/gsvloader-demo/master/geojson/Bangkok-districts.geojson"
    geojson_path = "bangkok_districts.geojson"

    # ดาวน์โหลดครั้งเดียว
    @st.cache_data
    def load_geojson():
        r = requests.get(geojson_url)
        with open(geojson_path, "wb") as f:
            f.write(r.content)
        with open(geojson_path, "r", encoding="utf-8") as f:
            return json.load(f)

    bangkok_geojson = load_geojson()
    # เติม avg_duration_hr ลงใน GeoJSON
    duration_map = dict(zip(df_avg["dname"], df_avg["avg_duration_hr"]))

    for feature in bangkok_geojson["features"]:
        dname = feature["properties"]["dname"]
        feature["properties"]["avg_duration_hr"] = round(duration_map.get(dname, 0), 2) if dname in duration_map else None


    # สร้างแผนที่
    m = folium.Map(location=[13.7563, 100.5018], zoom_start=11, width="100%", height="600px")

    Choropleth(
        geo_data=bangkok_geojson,
        data=df_avg,
        columns=["dname", "avg_duration_hr"],
        key_on="feature.properties.dname",
        fill_color="YlOrRd",
        fill_opacity=0.7,
        line_opacity=0.3,
        legend_name="เวลาเฉลี่ยในการแก้ปัญหา (ชั่วโมง)",
        threshold_scale=[0, 5, 10, 15, 20, 25, 30]
    ).add_to(m)

    folium.GeoJson(
        bangkok_geojson,
        style_function=lambda feature: {
            "fillColor": "black" if feature["properties"]["dname"] not in df_avg["dname"].values else "transparent",
            "color": "gray",
            "weight": 0.5,
            "fillOpacity": 0.5
        },
        tooltip=folium.GeoJsonTooltip(
        fields=["dname", "avg_duration_hr"],
        aliases=["เขต:", "เวลาเฉลี่ย (ชม.):"],
        localize=True
    )
    ).add_to(m)


    # แสดงผล
    st.header("Average Time to Resolve Issues by District")
    m.save("map.html")
    with open("map.html", "r", encoding="utf-8") as f:
        map_html = f.read()

    components.html(map_html, height=650, scrolling=False)





