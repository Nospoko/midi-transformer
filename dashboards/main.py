import streamlit as st

from dashboards.model_review import main as model_review
from dashboards.hf_dataset_review import main as hf_datasets_review


def main():
    options = ["hf_datasets_review", "model_review"]
    display_mode = st.selectbox(label="display mode", options=options)

    match display_mode:
        case "hf_datasets_review":
            hf_datasets_review()
        case "model_review":
            model_review()
