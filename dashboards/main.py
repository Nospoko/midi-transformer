import streamlit as st
from dotenv import load_dotenv

from dashboards.run_eval import main as run_eval
from dashboards.gpt_review import main as gpt_review
from dashboards.browse_generated import main as browse_generated
from dashboards.hf_dataset_review import main as hf_datasets_review
from dashboards.augmentation_review import main as augmentation_review
from dashboards.midi_dataset_review import main as midi_dataset_review

load_dotenv()


def main():
    options = [
        "browse generated",
        "gpt_review",
        "hf_datasets_review",
        "augmentation_review",
        "midi_dataset_review",
        "evaluation",
    ]
    display_mode = st.selectbox(label="display mode", options=options)

    match display_mode:
        case "hf_datasets_review":
            hf_datasets_review()
        case "gpt_review":
            gpt_review()
        case "augmentation_review":
            augmentation_review()
        case "midi_dataset_review":
            midi_dataset_review()
        case "evaluation":
            run_eval()
        case "browse generated":
            browse_generated()


if __name__ == "__main__":
    main()
