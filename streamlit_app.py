import os
import pandas as pd
import streamlit as st
import tempfile

# Set Streamlit page config
st.set_page_config(page_title="TriMark MMM", page_icon=":male_mage:")

# App title
st.title(":male_mage: TriMark MMM")

with st.sidebar:
    # Timeframe selector
    timeframe = st.date_input(
        "Select Timeframe",
        help="Choose a date range for the analysis (start and end dates)",
    )
    
    # Budget input
    budget = st.text_input("Selected Timeframe Budget")
    "Enter budget (e.g., 200000)"

    # File uploader for CSV and XLSX
    data_files = st.file_uploader("Upload your data files", accept_multiple_files=True, type=["csv", "xlsx"])
    add_data_files = st.session_state.get("add_data_files", [])
    for data_file in data_files:
        file_name = data_file.name
        if file_name in add_data_files:
            continue
        try:
            if not budget:
                st.error("Please enter your budget")
                st.stop()
            temp_file_name = None
            with tempfile.NamedTemporaryFile(mode="wb", delete=False, prefix=file_name) as f:
                f.write(data_file.getvalue())
                temp_file_name = f.name
            if temp_file_name:
                st.markdown(f"Adding {file_name} to knowledge base...")
                # Example processing logic for CSV and XLSX
                if file_name.endswith(".csv"):
                    df = pd.read_csv(temp_file_name)
                elif file_name.endswith(".xlsx"):
                    df = pd.read_excel(temp_file_name)
                # Add your specific logic here for handling the uploaded data
                st.write(df.head())  # Display first few rows as an example
                add_data_files.append(file_name)
                os.remove(temp_file_name)
            st.session_state.messages.append({"role": "assistant", "content": f"Added {file_name} to model!"})
        except Exception as e:
            st.error(f"Error adding {file_name} to model: {e}")
            st.stop()
    st.session_state["add_data_files"] = add_data_files
