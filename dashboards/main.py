import streamlit as st

from dashboards.gpt_review import main as gpt_review
from dashboards.hf_dataset_review import main as hf_datasets_review


def main():
    options = ["gpt_review", "hf_datasets_review"]
    display_mode = st.selectbox(label="display mode", options=options)

    match display_mode:
        case "hf_datasets_review":
            hf_datasets_review()
        case "gpt_review":
            gpt_review()


if __name__ == "__main__":
    main()
