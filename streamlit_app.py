import os
import pandas as pd
import streamlit as st

# Set Streamlit page config
st.set_page_config(page_title="The Reef", page_icon=":male_mage:")

# App title
st.title(":male_mage: TriMark MMM")

with st.sidebar:
    budget = st.text_input("Selected Timeframe Budget")

    pdf_files = st.file_uploader("Upload your PDF files", accept_multiple_files=True, type="pdf")
    add_pdf_files = st.session_state.get("add_pdf_files", [])
    for pdf_file in pdf_files:
        file_name = pdf_file.name
        if file_name in add_pdf_files:
            continue
        try:
            if not st.session_state.api_key:
                st.error("Please enter your OpenAI API Key")
                st.stop()
            temp_file_name = None
            with tempfile.NamedTemporaryFile(mode="wb", delete=False, prefix=file_name, suffix=".pdf") as f:
                f.write(pdf_file.getvalue())
                temp_file_name = f.name
            if temp_file_name:
                st.markdown(f"Adding {file_name} to knowledge base...")
                app.add(temp_file_name, data_type="pdf_file")
                st.markdown("")
                add_pdf_files.append(file_name)
                os.remove(temp_file_name)
            st.session_state.messages.append({"role": "assistant", "content": f"Added {file_name} to knowledge base!"})
        except Exception as e:
            st.error(f"Error adding {file_name} to knowledge base: {e}")
            st.stop()
    st.session_state["add_pdf_files"] = add_pdf_files
