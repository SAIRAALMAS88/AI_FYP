# Import necessary libraries
import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px
from ydata_profiling import ProfileReport
import tempfile
import os
from together import Together
import pdfplumber

# Set page config
st.set_page_config(
    page_title="AI-Powered Data Insights",
    page_icon="📊",
    layout="wide"
)

# Initialize Together API
try:
    together_api = st.secrets.get("TOGETHER_API_KEY", "76d4ee171011eb38e300cee2614c365855cd744e64282a8176cc178592aea8ce")
    client = Together(api_key=together_api)
except TypeError:
    try:
        client = Together()
        client.api_key = together_api
    except Exception as e:
        st.error(f"Failed to initialize Together AI client: {e}")
        st.stop()

def call_llama2(prompt):
    """Function to call the Together AI LLama2 model"""
    try:
        response = client.chat.completions.create(
            model="meta-llama/Llama-3.3-70B-Instruct-Turbo-Free",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=512,
            temperature=0.2,
            top_k=50,
            repetition_penalty=1,
            stop=["<❘end❘of❘sentence❘>"],
            top_p=0.7,
            stream=False
        )
        if hasattr(response, 'choices') and response.choices:
            return response.choices[0].message.content
        return "No response from AI."
    except Exception as e:
        return f"AI Error: {str(e)}"

def read_pdf(file):
    """Extract text from PDF with error handling"""
    try:
        with pdfplumber.open(file) as pdf:
            return "\n".join(page.extract_text() for page in pdf.pages if page.extract_text())
    except Exception as e:
        st.error(f"PDF reading error: {e}")
        return None

# UI Components
st.title("📊 AI-Powered Data Insights & Visualization Assistant")
uploaded_file = st.file_uploader(
    "Choose a file (CSV, Excel, PDF)",
    type=["csv", "xlsx", "pdf"]
)

if uploaded_file is not None:
    file_extension = uploaded_file.name.split('.')[-1].lower()
    data_frame = None
    
    try:
        if file_extension == 'csv':
            data_frame = pd.read_csv(uploaded_file)
        elif file_extension == 'xlsx':
            data_frame = pd.read_excel(uploaded_file)
        elif file_extension == 'pdf':
            pdf_text = read_pdf(uploaded_file)
            if pdf_text:
                with st.expander("📄 Extracted PDF Content"):
                    st.text(pdf_text)
                
                if st.button("Analyze PDF Content"):
                    analysis_prompt = f"Analyze this document and provide a summary of key points:\n{pdf_text[:10000]}"
                    analysis = call_llama2(analysis_prompt)
                    st.markdown("### 📝 Document Analysis")
                    st.write(analysis)
            # Skip visualization for PDFs by not setting data_frame
    except Exception as e:
        st.error(f"Error processing file: {e}")
        st.stop()

    # Only show data analysis if we have a dataframe (CSV/Excel)
    if data_frame is not None:
        # Basic Info Section
        st.success(f"✅ Successfully loaded {uploaded_file.name} ({data_frame.shape[0]} rows, {data_frame.shape[1]} columns)")
        
        with st.expander("🔍 Dataset Preview"):
            st.dataframe(data_frame.head(), use_container_width=True)
            
            if st.checkbox("Show random samples"):
                sample_size = st.slider("Sample size", 1, 100, 10)
                st.dataframe(data_frame.sample(sample_size), use_container_width=True)

        # Data Profiling Section
        st.subheader("📈 Automated Data Profiling")
        if st.button("Generate Full Profile Report"):
            with st.spinner("Generating comprehensive profile..."):
                try:
                    profile = ProfileReport(data_frame, title="Dataset Profile")
                    
                    with tempfile.NamedTemporaryFile(delete=False, suffix=".html") as tmpfile:
                        profile.to_file(tmpfile.name)
                        with open(tmpfile.name, "r", encoding="utf-8") as f:
                            html_content = f.read()
                    
                    st.components.v1.html(html_content, width=1000, height=1200, scrolling=True)
                    
                    with open(tmpfile.name, "rb") as f:
                        st.download_button(
                            "Download Full Report",
                            f.read(),
                            "data_profile.html",
                            "text/html"
                        )
                    os.unlink(tmpfile.name)
                except Exception as e:
                    st.error(f"Profile generation failed: {e}")

        # Visualization Section
        st.subheader("📊 Interactive Visualizations")
        vis_col1, vis_col2 = st.columns(2)
        
        with vis_col1:
            numeric_cols = data_frame.select_dtypes(include=['number']).columns.tolist()
            cat_cols = data_frame.select_dtypes(include=['object', 'category']).columns.tolist()
            
            plot_type = st.selectbox(
                "Select visualization type",
                ["Histogram", "Box Plot", "Scatter Plot", "Bar Chart", "Line Chart"]
            )
            
            x_axis = st.selectbox("X-axis", data_frame.columns)
            y_axis = st.selectbox("Y-axis", numeric_cols if numeric_cols else [None])

        with vis_col2:
            color_col = st.selectbox("Color by (optional)", [None] + cat_cols)
            facet_col = st.selectbox("Facet by (optional)", [None] + cat_cols)
            
            if st.button("Generate Visualization"):
                try:
                    fig = None
                    if plot_type == "Histogram":
                        fig = px.histogram(data_frame, x=x_axis, color=color_col, facet_col=facet_col)
                    elif plot_type == "Box Plot":
                        fig = px.box(data_frame, x=x_axis, y=y_axis, color=color_col)
                    elif plot_type == "Scatter Plot" and y_axis:
                        fig = px.scatter(data_frame, x=x_axis, y=y_axis, color=color_col)
                    elif plot_type == "Bar Chart":
                        fig = px.bar(data_frame, x=x_axis, y=y_axis if y_axis else None, color=color_col)
                    elif plot_type == "Line Chart" and y_axis:
                        fig = px.line(data_frame, x=x_axis, y=y_axis, color=color_col)
                    
                    if fig:
                        st.plotly_chart(fig, use_container_width=True)
                    else:
                        st.warning("Could not generate the selected visualization with current parameters")
                except Exception as e:
                    st.error(f"Visualization error: {e}")

        # AI Analysis Section
        st.subheader("🤖 AI-Powered Analysis")
        
        tab1, tab2 = st.tabs(["Automated EDA Summary", "Ask Questions"])
        
        with tab1:
            if st.button("Generate AI Summary"):
                with st.spinner("Analyzing data with AI..."):
                    eda_prompt = f"""
                    Analyze this dataset and provide insights:
                    - Columns: {data_frame.columns.tolist()}
                    - Sample data: {data_frame.head().to_dict()}
                    - Null values: {data_frame.isnull().sum().to_dict()}
                    - Data types: {data_frame.dtypes.astype(str).to_dict()}
                    
                    Provide:
                    1. Data quality assessment
                    2. Interesting patterns
                    3. Recommended visualizations
                    4. Potential data issues
                    """
                    response = call_llama2(eda_prompt)
                    st.markdown(response)
        
        with tab2:
            user_question = st.text_area("Ask anything about your data")
            if user_question and st.button("Get Answer"):
                with st.spinner("Thinking..."):
                    qa_prompt = f"""
                    Dataset info:
                    - Columns: {data_frame.columns.tolist()}
                    - Sample: {data_frame.head().to_dict()}
                    - Nulls: {data_frame.isnull().sum()}
                    
                    Question: {user_question}
                    
                    Answer concisely with relevant statistics if applicable.
                    """
                    answer = call_llama2(qa_prompt)
                    st.markdown(f"**Answer:** {answer}")

# Footer
st.markdown("---")
st.markdown("""
    <div style="text-align: center;">
        <p>AI-Powered Data Insights Assistant</p>
    </div>
""", unsafe_allow_html=True)
