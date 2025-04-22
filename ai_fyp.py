# Import necessary libraries 
import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px
from ydata_profiling import ProfileReport
import tempfile
import os
import together
import pdfplumber

# Initialize Together API
together_api = "76d4ee171011eb38e300cee2614c365855cd744e64282a8176cc178592aea8ce"
together.api_key = together_api

def call_llama2(prompt):
    try:
        response = together.chat.completions.create(
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
