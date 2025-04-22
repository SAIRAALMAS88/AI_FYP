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

# Check for required dependencies
try:
    import openpyxl
except ImportError:
    st.warning("The 'openpyxl' package is required for Excel file support. Installing now...")
    import subprocess
    import sys
    subprocess.check_call([sys.executable, "-m", "pip", "install", "openpyxl"])
    import openpyxl

# Set page config
st.set_page_config(
    page_title="AI-Powered Data Insights",
    page_icon="📊",
    layout="wide"
)

# Initialize Together API - CORRECT INITIALIZATION METHOD
try:
    together_api = st.secrets.get("TOGETHER_API_KEY", "your-api-key-here")
    client = Together()
    client.api_key = together_api  # Correct way to set API key in current version
except Exception as e:
    st.error(f"Failed to initialize Together AI client: {e}")
    st.stop()

def call_llama2(prompt):
    """Function to call the Together AI LLama2 model with improved error handling"""
    try:
        response = client.chat.completions.create(
            model="meta-llama/Llama-3-70b-chat-hf",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=1024,
            temperature=0.3,
            top_k=50,
            repetition_penalty=1,
            stop=["<|endoftext|>"],
            top_p=0.7,
            stream=False
        )
        if hasattr(response, 'choices') and response.choices:
            return response.choices[0].message.content
        return "No response from AI."
    except Exception as e:
        st.error(f"AI API Error: {str(e)}")
        return None

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
            data_frame = pd.read_excel(uploaded_file, engine='openpyxl')
        elif file_extension == 'pdf':
            pdf_text = read_pdf(uploaded_file)
            if pdf_text:
                with st.expander("📄 Extracted PDF Content"):
                    st.text(pdf_text[:5000] + ("..." if len(pdf_text) > 5000 else ""))
                
                if st.button("Analyze PDF Content"):
                    with st.spinner("Analyzing document..."):
                        analysis_prompt = f"""Please analyze this document and provide:
                        1. A concise summary of key points
                        2. Main topics covered
                        3. Any notable patterns or insights

                        Document content:\n{pdf_text[:15000]}"""
                        analysis = call_llama2(analysis_prompt)
                        if analysis:
                            st.markdown("### 📝 Document Analysis")
                            st.write(analysis)
            # Skip visualization for PDFs
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
                    profile = ProfileReport(data_frame, title="Dataset Profile", explorative=True)
                    
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
                    # Prepare dataset sample
                    sample_data = data_frame.head(3).to_dict(orient='records')
                    
                    eda_prompt = f"""Analyze this dataset and provide a structured report:
                    
                    Dataset Overview:
                    - Shape: {data_frame.shape}
                    - Columns: {list(data_frame.columns)}
                    - Sample Rows: {sample_data}
                    
                    Please provide:
                    1. Data Quality Assessment (missing values, duplicates)
                    2. Statistical Summary (for numeric columns)
                    3. Interesting Patterns/Observations
                    4. Recommendations for:
                       - Data Cleaning
                       - Further Analysis
                       - Potential Visualizations
                    
                    Keep the response concise and structured with clear headings."""
                    
                    response = call_llama2(eda_prompt)
                    if response:
                        st.markdown("### 📝 AI-Generated EDA Summary")
                        st.markdown(response)
                    else:
                        st.error("Failed to generate EDA summary")
        
        with tab2:
            user_question = st.text_area("Ask anything about your data", height=100)
            if user_question and st.button("Get Answer"):
                with st.spinner("Analyzing your question..."):
                    # Prepare context
                    context = {
                        "columns": list(data_frame.columns),
                        "dtypes": data_frame.dtypes.astype(str).to_dict(),
                        "sample": data_frame.head(3).to_dict(orient='records'),
                        "null_counts": data_frame.isnull().sum().to_dict()
                    }
                    
                    qa_prompt = f"""You are a data analyst assistant. Answer the following question about the dataset:
                    
                    Question: {user_question}
                    
                    Dataset Context:
                    - Columns: {context['columns']}
                    - Data Types: {context['dtypes']}
                    - Sample Data: {context['sample']}
                    - Null Values Count: {context['null_counts']}
                    
                    Provide:
                    1. A clear answer to the question
                    2. Relevant statistics if applicable
                    3. Any caveats or limitations in the data
                    4. Suggestions for further analysis if relevant
                    
                    If the question cannot be answered with the available data, explain why."""
                    
                    answer = call_llama2(qa_prompt)
                    if answer:
                        st.markdown("### 🤖 Analysis Results")
                        st.markdown(answer)
                    else:
                        st.error("Failed to
