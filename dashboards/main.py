import streamlit as st
from dotenv import load_dotenv

from dashboards.gpt_review import main as gpt_review
from dashboards.browse_generated import main as browse_generated
from dashboards.augmentation_review import main as augmentation_review
from dashboards.midi_dataset_review import main as midi_dataset_review
from dashboards.hf_midi_dataset_review import main as hf_datasets_review
from dashboards.extracted_voice_gpt_review import main as extracted_voice_gpt_review
from dashboards.subsequence_dataset_review import main as subsequense_dataset_review
from dashboards.hf_subsequence_dataset_review import main as hf_subsequence_dataset_review

load_dotenv()


def main():
    options = [
        "gpt_review",
        "extracted_voice_gpt_review",
        "browse generated",
        "hf_midi_datasets_review",
        "hf_subsequence_datasets_review",
        "augmentation_review",
        "midi_dataset_review",
        "subsequence_dataset_review",
        "evaluation",
    ]
    display_mode = st.selectbox(label="display mode", options=options)

    match display_mode:
        case "hf_midi_datasets_review":
            hf_datasets_review()
        case "hf_subsequence_datasets_review":
            hf_subsequence_dataset_review()
        case "gpt_review":
            gpt_review()
        case "extracted_voice_gpt_review":
            extracted_voice_gpt_review()
        case "augmentation_review":
            augmentation_review()
        case "midi_dataset_review":
            midi_dataset_review()
        case "browse generated":
            browse_generated()
        case "subsequence_dataset_review":
            subsequense_dataset_review()


if __name__ == "__main__":
    main()
