from typing import Any

import yaml
import torch
import streamlit as st
from datasets import Dataset, load_dataset


def select_part_dataset(midi_dataset: Dataset) -> Dataset:
    """
    Allows the user to select a part of the dataset based on composer and title.

    Parameters:
        midi_dataset (Dataset): The MIDI dataset to select from.

    Returns:
        Dataset: The selected part of the dataset.
    """
    source_df = midi_dataset.to_pandas()
    source_df["source"] = source_df["source"].map(lambda source: yaml.safe_load(source))
    source_df["composer"] = [source["composer"] for source in source_df.source]
    source_df["title"] = [source["title"] for source in source_df.source]

    composers = source_df.composer.unique()
    selected_composer = st.selectbox(
        "Select composer",
        options=composers,
        index=3,
    )

    ids = source_df.composer == selected_composer
    piece_titles = source_df[ids].title.unique()
    selected_title = st.selectbox("Select title", options=piece_titles)

    ids = (source_df.composer == selected_composer) & (source_df.title == selected_title)
    part_df = source_df[ids]
    part_dataset = midi_dataset.select(part_df.index.values)

    return part_dataset


def load_checkpoint(checkpoint_path: str, device: str):
    checkpoint = torch.load(checkpoint_path, map_location=device)
    return checkpoint


@st.cache_data
def load_hf_dataset(dataset_path: str, dataset_split: str):
    dataset = load_dataset(
        dataset_path,
        split=dataset_split,
        trust_remote_code=True,
        num_proc=8,
    )
    return dataset


def select_generation_parameters() -> (
    tuple[
        bool,
        dict[str, Any],
        float,
        float,
    ]
):
    st.header("Generation Parameters")
    with st.form("generate_parameters"):
        col1, col2 = st.columns(2)
        with col1:
            temperature = st.number_input("Temperature", value=1.0, help="Controls randomness in generation")
            max_new_tokens = st.number_input(
                "Max New Tokens",
                value=1024,
                help="Maximum number of new tokens to generate",
            )
        with col2:
            prompt_duration = st.number_input(
                "Whole prompt duration",
                value=10.0,
                help="Duration of the whole prompt for iterative generation",
            )
            prompt_creation_time_step = st.number_input(
                "Prompt Creation Time Step",
                value=10.0,
                help="Time step for creating prompts",
            )
            task_options = ["bass_prediction", "reverse_bass_prediction", "next_token_prediction"]
            task = st.selectbox(label="task", options=task_options)
            prompt_context_duration = st.number_input(
                "Prompt Context Duration",
                min_value=1.0,
                max_value=30.0,
                value=10.0,
                help="Duration of the prompt context in seconds",
            )

            target_context_duration = 0.0
            time_step = 0.0
            if task != "next_token_prediction":
                target_context_duration = st.number_input(
                    "Target Context Duration",
                    min_value=0.0,
                    max_value=30.0,
                    value=0.0,
                    help="Duration of the bass context in seconds",
                )
                time_step = st.number_input(
                    "Generation Time Step",
                    min_value=0.0,
                    max_value=30.0,
                    value=10.0,
                    help="Time step for iterative generation",
                )

        run = st.form_submit_button("Generate Notes")

    generation_parameters = {
        "temperature": temperature,
        "max_new_tokens": max_new_tokens,
        "prompt_context_duration": prompt_context_duration,
        "target_context_duration": target_context_duration,
        "time_step": time_step,
        "task": task,
    }

    st.image("dashboards/img/iterative_generation.png")
    return run, generation_parameters, prompt_duration, prompt_creation_time_step
