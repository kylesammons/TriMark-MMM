import os
import pandas as pd
import streamlit as st
import tempfile

# Set Streamlit page config
st.set_page_config(page_title="TriMark MMM", page_icon=":male_mage:")

# App title
st.title(":male_mage: TriMark MMM")

# Sidebar for inputs
with st.sidebar:
    st.title("Select Timeframe")
    startdate = st.date_input(
        "Start Date",
        help="Choose a start date for the date range that you want media mix predictions/recommendations for",
    )
    enddate = st.date_input(
        "End Date",
        help="Choose an end date for the date range that you want media mix predictions/recommendations for",
    )

    st.title("Input Budget")
    budget = st.text_input(
        "Total Budget",
        help="Input total planned budget for all channels included in your data file for the selected time frame",
    )
    
    st.title("Load Data")
    data_files = st.file_uploader(
        "Upload your data files", accept_multiple_files=True, type=["csv", "xlsx"]
    )

# Initialize session state for tracking files, data, and training
if "add_data_files" not in st.session_state:
    st.session_state["add_data_files"] = []

if "df" not in st.session_state:
    st.session_state["df"] = None

if "response_variable" not in st.session_state:
    st.session_state["response_variable"] = None

if "model_trained" not in st.session_state:
    st.session_state["model_trained"] = False

if "submitted" not in st.session_state:
    st.session_state["submitted"] = False


# Function to process the uploaded file
def process_file(data_file):
    file_name = data_file.name
    if file_name in st.session_state["add_data_files"]:
        return None

    try:
        if not budget:
            st.error("Please enter your budget")
            return None

        temp_file_name = None
        with tempfile.NamedTemporaryFile(mode="wb", delete=False, prefix=file_name) as f:
            f.write(data_file.getvalue())
            temp_file_name = f.name

        if temp_file_name:
            if file_name.endswith(".csv"):
                df = pd.read_csv(temp_file_name)
            elif file_name.endswith(".xlsx"):
                df = pd.read_excel(temp_file_name)

            # Check if dataframe is empty
            if df.empty:
                st.error("Uploaded file is empty. Please upload a valid file.")
                return None

            # Store dataframe in session state
            st.session_state["df"] = df

            # Mark the file as processed
            st.session_state["add_data_files"].append(file_name)

            return df
    except Exception as e:
        st.error(f"Error adding `{file_name}` to model: {e}")
        return None

# Main workflow
if data_files:
    for data_file in data_files:
        df = process_file(data_file)
        if df is None:
            continue
        
        # Show the first few rows of the uploaded file
        st.dataframe(df.head(50), height=400)

        # Ensure that columns are available before creating the form
        if df.columns.size > 0:
            st.success(f"Added `{data_file.name}` to model!")

            # Create a form for the Response Variable selection
            with st.form(key="response_var_form"):
                response_var = st.selectbox(
                    "Response Variable",
                    options=df.columns.tolist(),  # Use column names from the dataframe
                    help="Select the response variable for analysis"
                )
                
                model_type = st.selectbox(
                    "Model Type",
                    options=["Carryover", "Adstock", "Hill Adstock"],
                    help="Select the type of model for media mix analysis"
                )

                # Submit button for the form
                submit_button = st.form_submit_button(label="Run Model")

            # Handle form submission
            if submit_button and not st.session_state["submitted"]:
                st.session_state["submitted"] = True  # Track form submission state

                with st.spinner("Training the MMM model..."):
                    # Simulate model training (replace with actual model logic)
                    st.session_state["response_variable"] = response_var
                    st.session_state["model_trained"] = True
                    st.success(f"Model trained with `{response_var}` as the response variable!")

                st.session_state["submitted"] = False  # Reset form submission state after completion

# After form submission, show the trained model info if available
if st.session_state.get("model_trained", False):
    st.success("Model has already been trained. You can view or analyze the results here.")
