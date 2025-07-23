import streamlit as st
import requests
import json
from datetime import datetime, timedelta
import base64
from PIL import Image
import io

st.set_page_config(
    page_title="Phytoplankton Research Assistant", 
    page_icon="🌊",
    layout="wide"
)

# Custom CSS
st.markdown("""
<style>
.main-header {
    text-align: center;
    padding: 1rem 0;
    background: linear-gradient(90deg, #1e3c72 0%, #2a5298 100%);
    color: white;
    border-radius: 10px;
    margin-bottom: 2rem;
}
.feature-box {
    background: #f0f8ff;
    padding: 1rem;
    border-radius: 8px;
    border-left: 4px solid #2a5298;
    margin: 1rem 0;
}
.nasa-info {
    background: #e8f4fd;
    padding: 1rem;
    border-radius: 8px;
    border: 1px solid #bee5eb;
}
</style>
""", unsafe_allow_html=True)

# Header
st.markdown('<div class="main-header"><h1>🌿 Phytoplankton Research Assistant</h1><p>Analyze coastal satellite images and ask research questions</p></div>', unsafe_allow_html=True)

# Sidebar for configuration
with st.sidebar:
    st.header("⚙️ Configuration")
    
    # NASA settings
    st.subheader("🛰️ NASA Satellite Images")
    use_nasa = st.checkbox("Auto-fetch NASA satellite image", help="Automatically fetch a satellite image if none is uploaded")
    
    if use_nasa:
        col1, col2 = st.columns(2)
        with col1:
            custom_lat = st.number_input("Latitude", value=None, placeholder="Auto", help="Leave empty for random coastal location")
            custom_lon = st.number_input("Longitude", value=None, placeholder="Auto", help="Leave empty for random coastal location")
        
        with col2:
            date_option = st.selectbox("Date", ["Recent (Auto)", "Custom"])
            if date_option == "Custom":
                custom_date = st.date_input(
                    "Select date",
                    value=datetime.now() - timedelta(days=7),
                    max_value=datetime.now() - timedelta(days=1)
                )
            else:
                custom_date = None
    
    # Predefined locations
    st.subheader("📍 Quick Locations")
    location_options = {
        "Custom": (None, None),
        "San Francisco Bay": (37.7749, -122.4194),
        "Miami Coast": (25.7617, -80.1918),
        "New York Coast": (40.7128, -74.0060),
        "Los Angeles Coast": (34.0522, -118.2437),
        "Seattle Coast": (47.6062, -122.3321)
    }
    
    selected_location = st.selectbox("Choose location", list(location_options.keys()))
    if selected_location != "Custom":
        custom_lat, custom_lon = location_options[selected_location]

# Main interface
col1, col2 = st.columns([1, 1])

with col1:
    st.subheader("📤 Upload Image (Optional)")
    uploaded_image = st.file_uploader(
        "Upload a coastal satellite image", 
        type=["jpg", "jpeg", "png"],
        help="Upload your own satellite or coastal image, or leave empty to auto-fetch from NASA"
    )
    
    if uploaded_image:
        st.image(uploaded_image, caption="Uploaded Image", use_column_width=True)
        st.markdown('<div class="feature-box">✅ Using your uploaded image</div>', unsafe_allow_html=True)
    elif use_nasa:
        st.markdown('<div class="nasa-info">🛰️ Will auto-fetch NASA satellite image when you ask a question</div>', unsafe_allow_html=True)

with col2:
    st.subheader("❓ Ask Your Question")
    
    # Sample questions
    sample_questions = [
        "What phytoplankton species can you identify in this image?",
        "Are there any signs of algal blooms in this satellite image?",
        "What is the frequency of Noctiluca blooms in different months?",
        "Describe the ocean color patterns in this coastal area",
        "What environmental factors affect phytoplankton distribution?",
        "How do satellite images help in phytoplankton research?"
    ]
    
    selected_sample = st.selectbox("Or choose a sample question:", ["Custom question..."] + sample_questions)
    
    if selected_sample != "Custom question...":
        question = st.text_area("Your question:", value=selected_sample, height=100)
    else:
        question = st.text_area("Your question:", placeholder="e.g., What phytoplankton can you see in this image?", height=100)

# Analysis button
if st.button("🔍 Analyze", type="primary", use_container_width=True):
    if not question.strip():
        st.error("Please enter a question!")
    else:
        with st.spinner("🤖 Analyzing... This may take a moment"):
            try:
                # Prepare request
                files = {}
                data = {
                    "question": question,
                    "use_nasa": use_nasa
                }
                
                # Add image if uploaded
                if uploaded_image:
                    files["image"] = uploaded_image.getvalue()
                
                # Add NASA parameters if using NASA
                if use_nasa:
                    if custom_lat is not None:
                        data["lat"] = custom_lat
                    if custom_lon is not None:
                        data["lon"] = custom_lon
                    if custom_date is not None:
                        data["date"] = custom_date.strftime("%Y-%m-%d")
                
                # Make request
                response = requests.post(
                    "http://localhost:8000/analyze", 
                    files=files if files else None,
                    data=data,
                    timeout=60
                )
                
                if response.status_code == 200:
                    result = response.json()
                    
                    # Display results
                    st.success("✅ Analysis Complete!")
                    
                    # Main answer
                    st.subheader("🎯 Answer:")
                    st.markdown(f"**{result['answer']}**")
                    
                    # Show NASA image if used
                    if result.get('nasa_used', False):
                        st.info("🛰️ Used NASA satellite image for analysis")
                    
                    # Expandable sections for detailed results
                    with st.expander("🖼️ Image Analysis Details", expanded=False):
                        if result.get('image_analysis'):
                            st.write(result['image_analysis'])
                        else:
                            st.write("No image analysis performed")
                    
                    with st.expander("📚 Knowledge Base Answer", expanded=False):
                        if result.get('rag_answer'):
                            st.write(result['rag_answer'])
                        else:
                            st.write("No knowledge base information found")
                    
                    with st.expander("📖 Source Context", expanded=False):
                        if result.get('context'):
                            st.text(result['context'])
                        else:
                            st.write("No context available")
                
                else:
                    st.error(f"❌ Error: {response.status_code}")
                    if response.text:
                        error_detail = response.json().get('detail', 'Unknown error')
                        st.error(f"Details: {error_detail}")
                        
            except requests.exceptions.Timeout:
                st.error("⏰ Request timed out. The analysis is taking longer than expected. Please try again.")
            except requests.exceptions.ConnectionError:
                st.error("🔌 Cannot connect to the backend server. Make sure the API server is running on localhost:8000")
            except Exception as e:
                st.error(f"❌ An unexpected error occurred: {str(e)}")

# NASA image preview section
st.markdown("---")
st.subheader("🛰️ NASA Satellite Image Preview")

col1, col2, col3 = st.columns([1, 1, 1])

with col2:
    if st.button("🌊 Fetch Random Coastal Image", use_container_width=True):
        with st.spinner("Fetching NASA image..."):
            try:
                params = {}
                if selected_location != "Custom":
                    lat, lon = location_options[selected_location]
                    params["lat"] = lat
                    params["lon"] = lon
                
                response = requests.get("http://localhost:8000/fetch-nasa-image", params=params)
                
                if response.status_code == 200:
                    result = response.json()
                    if result['success']:
                        # Display the fetched image
                        st.image(result['image'], caption="NASA Satellite Image", use_column_width=True)
                        st.success("🛰️ NASA image fetched successfully!")
                    else:
                        st.error("Failed to fetch NASA image")
                else:
                    st.error("Could not connect to NASA image service")
                    
            except Exception as e:
                st.error(f"Error fetching NASA image: {str(e)}")

# Footer with information
st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #666; padding: 1rem;">
    <p>🌊 This tool combines satellite image analysis with scientific literature to provide comprehensive answers about phytoplankton and coastal oceanography.</p>
    <p><strong>Features:</strong> BLIP-2 Image Analysis • RAG Knowledge Base • NASA Satellite Integration</p>
</div>
""", unsafe_allow_html=True)