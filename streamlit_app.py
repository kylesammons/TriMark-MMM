import os
import pandas as pd
import streamlit as st
import tempfile
import time
import seaborn as sns

# Set Streamlit page config
st.set_page_config(page_title="TriMark MMM", page_icon=":male_mage:")

# App title
st.title(":male_mage: TriMark MMM")

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

# Sidebar inputs
with st.sidebar:
    st.title("Select Timeframe")
    startdate = st.date_input("Start Date")
    enddate = st.date_input("End Date")
    budget = st.text_input("Total Budget")
    data_files = st.file_uploader("Upload your data files", accept_multiple_files=True, type=["csv", "xlsx"])

# Function to process uploaded file
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

# Main Workflow
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
                response_var = st.selectbox("Response Variable", options=df.columns.tolist())
                model_type = st.selectbox("Model Type", options=["Carryover", "Adstock", "Hill Adstock"])
                submit_button = st.form_submit_button(label="Run Model")

            # Handle form submission
            if submit_button and not st.session_state["submitted"]:
                st.session_state["submitted"] = True  # Mark form as submitted

                # Show spinner while simulating model training
                with st.spinner("Training the MMM model..."):
                    time.sleep(2)  # Simulating a time delay for model training
                    # Simulate model training logic
                    st.session_state["response_variable"] = response_var
                    st.session_state["model_trained"] = True

                    st.success(f"Model trained with `{response_var}` as the response variable!")

                st.session_state["submitted"] = False  # Reset form submission state

# After form submission, show the trained model info and the dashboard if available
if st.session_state.get("model_trained", False):
    st.success("Model has been trained. Below is the dashboard with the model results.")

    # Simple dashboard displaying simulated results
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.barplot(x=['TV', 'Radio', 'Online', 'Social'], y=[100, 80, 120, 90], ax=ax)  # Replace with model results
    ax.set_title('Media Spend Allocation')
    ax.set_ylabel('Budget Allocation')
    ax.set_xlabel('Channels')
    st.pyplot(fig)

    # Display some other model results (could be coefficients or insights)
    st.write("### Model Coefficients:")
    st.write({
        'TV': 0.45,
        'Radio': 0.30,
        'Online': 0.15,
        'Social': 0.10,
    })

    st.write("### Additional Insights:")
    st.write("The model predicts that TV spend has the highest effect on the response variable, followed by online marketing.")

    # You can add more visualizations or insights here based on your model results
