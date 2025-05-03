import streamlit as st
import pandas as pd
import pydeck as pdk
import plotly.express as px
from sklearn.cluster import DBSCAN
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