# Import required libraries
import os
import pandas as pd
import streamlit as st
import tempfile
import jax.numpy as jnp
from lightweight_mmm import lightweight_mmm, optimize_media, plot, preprocessing, utils
from datetime import datetime

# Set Streamlit page config
st.set_page_config(page_title="TriMark MMM", page_icon=":male_mage:")

# App title
st.title(":male_mage: TriMark MMM")

# Sidebar for inputs
with st.sidebar:
    st.title("Select Timeframe")
    # Start Date Timeframe selector
    startdate = st.date_input(
        "Start Date",
        help="Choose a start date for the date range that you want media mix predictions/recommendations for",
    )

    # End Date Timeframe selector
    enddate = st.date_input(
        "End Date",
        help="Choose an end date for the date range that you want media mix predictions/recommendations for",
    )

    st.title("Input Budget")
    # Budget input
    budget = st.text_input(
        "Total Budget",
        help="Input total planned budget for all channels included in your data file for the selected time frame",
    )
    
    st.title("Load Data")
    # File uploader for CSV and XLSX
    data_files = st.file_uploader(
        "Upload your data files", accept_multiple_files=True, type=["csv", "xlsx"]
    )

# Initialize session state for tracking files
if "add_data_files" not in st.session_state:
    st.session_state["add_data_files"] = []
if "mmm" not in st.session_state:
    st.session_state["mmm"] = None

# Display results in the main area
if data_files:
    for data_file in data_files:
        file_name = data_file.name
        if file_name in st.session_state["add_data_files"]:
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
                st.markdown(f"### Processing `{file_name}`")
                # Example processing logic for CSV and XLSX
                if file_name.endswith(".csv"):
                    df = pd.read_csv(temp_file_name)
                elif file_name.endswith(".xlsx"):
                    df = pd.read_excel(temp_file_name)
                
                # Check if dataframe is empty
                if df.empty:
                    st.error("Uploaded file is empty. Please upload a valid file.")
                    st.stop()

                # Show the first few rows of the uploaded file
                st.dataframe(df.head(50), height=400)

                # Mark the file as processed
                st.session_state["add_data_files"].append(file_name)

                # Ensure that columns are available before creating the form
                if df.columns.size > 0:
                    # Display success message first
                    st.success(f"Added `{file_name}` to model!")

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

                    # Run the MMM if the form is submitted
                    if submit_button:
                        try:
                            # Preprocessing and modeling logic
                            channels = [col for col in df.columns if col not in [response_var, 'Year', 'Month', 'Week']]
                            media_data = df[channels].to_numpy()
                            target = df[response_var].to_numpy()
                            costs = df[channels].sum().to_numpy()
                            target = target.astype(int)

                            data_size = media_data.shape[0]
                            split_point = data_size - 30

                            # Split data
                            media_data_train = media_data[:split_point, ...]
                            media_data_test = media_data[split_point:, ...]
                            target_train = target[:split_point].reshape(-1)

                            # Scale data
                            media_scaler = preprocessing.CustomScaler(divide_operation=jnp.mean)
                            target_scaler = preprocessing.CustomScaler(divide_operation=jnp.mean)
                            cost_scaler = preprocessing.CustomScaler(divide_operation=jnp.mean)

                            media_data_train = media_scaler.fit_transform(media_data_train)
                            target_train = target_scaler.fit_transform(target_train)
                            costs2 = cost_scaler.fit_transform(costs)

                            # Initialize and fit the model
                            mmm = lightweight_mmm.LightweightMMM(model_name="hill_adstock")
                            mmm.fit(
                                media=media_data_train,
                                media_prior=costs2,
                                target=target_train,
                                number_warmup=100,
                                number_samples=100,
                                number_chains=1,
                            )

                            # Store the model in session state
                            st.session_state["mmm"] = mmm

                            st.success("MMM successfully trained!")

                            # Display fit results
                            st.pyplot(plot.plot_model_fit(mmm, target_scaler=target_scaler))

                        except Exception as e:
                            st.error(f"Error running MMM: {e}")
                            st.stop()

                else:
                    st.error("No columns found in the uploaded file.")
                    st.stop()
                
        except Exception as e:
            st.error(f"Error adding `{file_name}` to model: {e}")
            st.stop()
