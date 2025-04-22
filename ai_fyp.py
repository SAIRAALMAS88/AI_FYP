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

# Initialize Together API
together_api = "76d4ee171011eb38e300cee2614c365855cd744e64282a8176cc178592aea8ce"
client = Together(api_key=together_api)

def call_llama2(prompt):
    try:
        response = client.chat.completions.create(
            model="meta-llama/Llama-3.3-70B-Instruct-Turbo-Free",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=None,
            temperature=0.3,
            top_k=50,
            repetition_penalty=1,
            stop=["<❘end❘of❘sentence❘>"],
            top_p=0.7,
            stream=False
        )
        if hasattr(response, 'choices') and response.choices:
            return response.choices[0].message.content
        else:
            return "No response from AI."
    except Exception as e:
        return f"AI Error: {e}"

# Set the title of the Streamlit app
st.title("AI-Powered Data Insights & Visualization Assistant")

# Add a file uploader to allow users to upload their dataset (CSV, Excel, PDF)
uploaded_file = st.file_uploader("Upload your dataset (CSV, Excel, or PDF)", type=["csv", "xlsx", "pdf"])

# Function to read PDF file
def read_pdf(file):
    with pdfplumber.open(file) as pdf:
        all_text = ""
        for page in pdf.pages:
            all_text += page.extract_text()
        return all_text

# Check if a file has been uploaded
if uploaded_file is not None:
    # Check file type
    file_extension = uploaded_file.name.split('.')[-1].lower()

    if file_extension == 'csv':
        # Load the dataset into a pandas DataFrame for CSV
        data_frame = pd.read_csv(uploaded_file)

    elif file_extension == 'xlsx':
        # Load the dataset into a pandas DataFrame for Excel
        data_frame = pd.read_excel(uploaded_file)

    elif file_extension == 'pdf':
        # Read the PDF file content (you may need to parse it further based on the layout)
        pdf_text = read_pdf(uploaded_file)
        st.write("### PDF Text Extracted")
        st.write(pdf_text)
        data_frame = None
        st.warning("PDF files are not structured like CSV or Excel, so visualization is not possible directly.")

    else:
        st.error("Unsupported file type.")

    # Check if a dataset is loaded (CSV/Excel)
    if data_frame is not None:
        # Display the first few rows of the dataset
        st.write("### Preview of the Dataset")
        st.dataframe(data_frame.head())

        # Allow users to view random samples of the dataset
        if st.checkbox("Show Random Samples"):
            sample_size = st.slider("Select sample size", 1, 100, 10)
            st.write(f"### Random {sample_size} Samples from the Dataset")
            st.dataframe(data_frame.sample(sample_size))

        # Display basic dataset information
        st.write("### Dataset Information")
        st.write(f"Number of Rows: {data_frame.shape[0]}")
        st.write(f"Number of Columns: {data_frame.shape[1]}")
        st.write("Column Names:")
        st.write(data_frame.columns.tolist())

        # Generate the Pandas Profiling report
        st.write("### Pandas Profiling Report")
        if st.button("Generate Pandas Profiling Report"):
            try:
                prof = ProfileReport(data_frame, minimal=True)
                st.write("Report generated successfully!")

                with tempfile.NamedTemporaryFile(delete=False, suffix=".html") as tmpfile:
                    prof.to_file(output_file=tmpfile.name)
                    tmpfile_path = tmpfile.name

                with open(tmpfile_path, "r", encoding="utf-8") as f:
                    html_content = f.read()
                st.components.v1.html(html_content, width=1000, height=1500, scrolling=True)

                with open(tmpfile_path, "rb") as f:
                    report_bytes = f.read()
                st.download_button(
                    label="Download Pandas Profiling Report",
                    data=report_bytes,
                    file_name="pandas_profiling_report.html",
                    mime="text/html"
                )

                os.unlink(tmpfile_path)
            except Exception as e:
                st.error(f"Error generating report: {e}")

        # Column Selection for Visualization
        st.write("### Column Selection for Visualization")
        selected_columns = st.multiselect("Select columns to visualize", data_frame.columns)

        if selected_columns:
            st.write(f"#### Visualizations for Selected Columns: {', '.join(selected_columns)}")

            for column in selected_columns:
                st.write(f"##### Column: {column}")

                if data_frame[column].dtype == "object" or data_frame[column].nunique() < 10:
                    st.write(f"**Pie Chart for {column}**")
                    fig = px.pie(data_frame, names=column, title=f"Pie Chart for {column}")
                    st.plotly_chart(fig)

                    st.write(f"**Bar Plot for {column}**")
                    fig = px.bar(data_frame[column].value_counts(), title=f"Bar Plot for {column}")
                    st.plotly_chart(fig)

                elif pd.api.types.is_numeric_dtype(data_frame[column]):
                    st.write(f"**Histogram for {column}**")
                    fig = px.histogram(data_frame, x=column, title=f"Histogram for {column}")
                    st.plotly_chart(fig)

                    st.write(f"**Boxplot for {column}**")
                    fig = px.box(data_frame, y=column, title=f"Boxplot for {column}")
                    st.plotly_chart(fig)

                    st.write(f"**Violin Plot for {column}**")
                    fig = px.violin(data_frame, y=column, title=f"Violin Plot for {column}")
                    st.plotly_chart(fig)

                elif pd.api.types.is_datetime64_any_dtype(data_frame[column]):
                    st.write(f"**Time Series Plot for {column}**")
                    fig = px.line(data_frame, x=column, y=data_frame.select_dtypes(include=["int64", "float64"]).columns[0], title=f"Time Series Plot for {column}")
                    st.plotly_chart(fig)
                else:
                    st.write(f"**Unsupported data type for {column}**")

        # Correlation Heatmap for Numeric Columns
        st.write("### Correlation Heatmap")
        numeric_columns = data_frame.select_dtypes(include=["int64", "float64"]).columns
        if not numeric_columns.empty and len(numeric_columns) >= 2:
            fig, ax = plt.subplots(figsize=(10, 8), dpi=100)
            sns.heatmap(
                data_frame[numeric_columns].corr(),
                annot=True,
                cmap="coolwarm",
                fmt=".2f",
                annot_kws={"size": 12},
                ax=ax
            )
            plt.xticks(rotation=45)
            plt.yticks(rotation=0)
            st.pyplot(fig)
        else:
            st.write("Not enough numeric columns for a correlation heatmap.")

        # Pairplot for Numeric Columns
        st.write("### Pairplot for Numeric Columns")
        if not numeric_columns.empty and len(numeric_columns) >= 2:
            numeric_columns_list = numeric_columns.tolist()
            default_columns = numeric_columns_list[:min(3, len(numeric_columns_list))]
            selected_pairplot_columns = st.multiselect("Select columns for pairplot", numeric_columns_list, default=default_columns)
            if selected_pairplot_columns:
                fig = sns.pairplot(data_frame[selected_pairplot_columns])
                st.pyplot(fig)

        # === AI-Powered EDA Summary ===
        st.write("## 📊 AI-Generated EDA Summary")
        if st.button("Generate EDA Summary"):
            eda_summary_prompt = f"""
The following EDA steps were performed on this dataset:
- Preview of dataset
- Null value check
- Data types summary
- Column-wise univariate visualizations (histograms, pie charts, boxplots, violin plots)
- Correlation heatmap
- Pairplot for numeric columns
- Pandas Profiling report was optionally generated

Please provide a concise summary of the dataset’s characteristics, highlight any potential data quality issues, and suggest further analysis or preprocessing steps.

Quick Stats:
- Columns: {data_frame.columns.tolist()}
- Nulls: {data_frame.isnull().sum().to_dict()}
- Dtypes: {data_frame.dtypes.astype(str).to_dict()}
"""

            eda_summary = call_llama2(eda_summary_prompt)
            st.markdown("### 📝 Summary:")
            st.markdown(eda_summary)

        # === AI-Powered Question Answering about Dataset ===
        st.write("## 🤖 Ask Questions About Your Dataset")
        user_question = st.text_input("Ask a question about the dataset")

        if user_question:
            # Prepare dataset sample and info
            df_head = data_frame.head(5).to_csv(index=False)
            df_schema = f"Column names and types: {data_frame.dtypes.astype(str).to_dict()}"
            null_info = f"Null values: {data_frame.isnull().sum().to_dict()}"

            question_prompt = f"""
I have the following dataset:
- Data: {df_head}
- Schema: {df_schema}
- Null info: {null_info}

Answer the following question: {user_question}
"""

            question_answer = call_llama2(question_prompt)
            st.markdown("### 🤖 Answer:")
            st.write(question_answer)

