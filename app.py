import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import plotly.express as px
import plotly.graph_objects as go

st.set_page_config(page_title="Wetland Hysteresis Analysis", layout="wide")

st.title("🌿 Functional Wetland Types: Hysteresis Clustering Demo")
st.markdown("""
This demo explores the concept of **Functional Hysteresis Types** by clustering synthetic wetland methane emission data.
The goal is to discover emergent patterns that transcend traditional wetland classifications.
""")

# Sidebar for controls
st.sidebar.header("Clustering Parameters")
n_clusters = st.sidebar.slider("Number of Clusters", 2, 5, 3)
n_wetlands = st.sidebar.slider("Number of Synthetic Wetlands", 10, 50, 25)

# Generate synthetic hysteresis data
@st.cache_data
def generate_synthetic_data(n_samples=25):
    np.random.seed(42)
    
    data = []
    wetland_types = ['Bog', 'Fen', 'Marsh', 'Swamp']
    regions = ['Alaska', 'Scandinavia', 'Canada', 'Siberia']
    
    for i in range(n_samples):
        wetland_type = np.random.choice(wetland_types)
        region = np.random.choice(regions)
        
        # Generate hysteresis features
        hysteresis_strength = np.random.normal(0.3, 0.15)  # H_A metric
        peak_lag = np.random.normal(25, 10)  # Days after temp peak
        spring_slope = np.random.normal(0.8, 0.3)
        fall_slope = np.random.normal(1.2, 0.4)
        asymmetry = fall_slope - spring_slope
        
        # Create some functional patterns
        if region in ['Alaska', 'Siberia']:
            hysteresis_strength += 0.1  # Cold regions have stronger hysteresis
        if wetland_type in ['Bog', 'Fen']:
            peak_lag += 5  # Peatlands have longer lags
            
        data.append({
            'wetland_id': f'W{i:03d}',
            'type': wetland_type,
            'region': region,
            'hysteresis_strength': max(0.1, hysteresis_strength),
            'peak_lag_days': max(5, peak_lag),
            'spring_slope': spring_slope,
            'fall_slope': fall_slope,
            'asymmetry': asymmetry
        })
    
    return pd.DataFrame(data)

# Main app
df = generate_synthetic_data(n_wetlands)

st.header("📊 Synthetic Wetland Dataset")
st.dataframe(df, use_container_width=True)

# Prepare features for clustering
features = ['hysteresis_strength', 'peak_lag_days', 'asymmetry']
X = df[features]

# Standardize features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Perform clustering
kmeans = KMeans(n_clusters=n_clusters, random_state=42)
df['cluster'] = kmeans.fit_predict(X_scaled)

# Visualization
col1, col2 = st.columns(2)

with col1:
    st.subheader("📈 Traditional Classification (By Wetland Type)")
    fig1 = px.scatter_3d(df, x='hysteresis_strength', y='peak_lag_days', z='asymmetry',
                        color='type', symbol='region',
                        title="Colored by Traditional Wetland Type",
                        labels={'hysteresis_strength': 'Hysteresis Strength (H_A)',
                               'peak_lag_days': 'Peak Lag (days)',
                               'asymmetry': 'Spring-Fall Asymmetry'})
    st.plotly_chart(fig1, use_container_width=True)

with col2:
    st.subheader("🎯 Functional Hysteresis Types (Clustering Result)")
    fig2 = px.scatter_3d(df, x='hysteresis_strength', y='peak_lag_days', z='asymmetry',
                        color=df['cluster'].astype(str), symbol='region',
                        title="Colored by Functional Hysteresis Cluster",
                        labels={'hysteresis_strength': 'Hysteresis Strength (H_A)',
                               'peak_lag_days': 'Peak Lag (days)',
                               'asymmetry': 'Spring-Fall Asymmetry'},
                        color_discrete_sequence=px.colors.qualitative.Bold)
    st.plotly_chart(fig2, use_container_width=True)

# Cluster analysis
st.header("🔍 Cluster Analysis")

# Show cross-tabulation
st.subheader("Cross-tabulation: Traditional vs Functional Types")
cross_tab = pd.crosstab(df['type'], df['cluster'])
st.dataframe(cross_tab.style.background_gradient(cmap='Blues'))

# Interpretation
st.subheader("💡 Interpretation")
st.markdown(f"""
The clustering analysis has identified **{n_clusters} functional hysteresis types** based on:
- **Hysteresis Strength** (H_A): Overall magnitude of the seasonal memory effect
- **Peak Lag**: How long after temperature peak methane peaks occur  
- **Asymmetry**: Difference between spring and fall emission rates

**Key Insight**: Notice how wetlands from different traditional classifications (Bog, Fen, Marsh) and different regions are grouped together in the same functional clusters. This suggests that underlying biophysical processes—not just taxonomy—determine methane emission patterns.
""")

# Show specific examples
st.subheader("🌍 Example Cross-Region Similarities")
cluster_examples = []
for cluster_id in df['cluster'].unique():
    cluster_df = df[df['cluster'] == cluster_id]
    if len(cluster_df) >= 2:
        example = cluster_df[['wetland_id', 'type', 'region', 'cluster']].head(2)
        cluster_examples.append(example)

if cluster_examples:
    examples_df = pd.concat(cluster_examples)
    st.dataframe(examples_df, use_container_width=True)
    st.caption("These wetlands from different regions and types show similar functional behavior.")

# Add download option
st.sidebar.header("Data Export")
if st.sidebar.button("Download Synthetic Data as CSV"):
    csv = df.to_csv(index=False)
    st.sidebar.download_button(
        label="Download CSV",
        data=csv,
        file_name="wetland_hysteresis_data.csv",
        mime="text/csv"
    )

st.sidebar.markdown("---")
st.sidebar.info("""
**About this Demo**: This demonstrates the research concept proposed in my cover letter to Dr. Yuan's lab, showing how unsupervised learning can reveal functional wetland types beyond traditional classifications.
""")
