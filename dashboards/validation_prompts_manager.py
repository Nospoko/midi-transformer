import json

import pandas as pd
import fortepyan as ff
import streamlit as st
import streamlit_pianoroll

from gpt2.generation import prepare_prompts
import data.database_manager as database_manager
import dashboards.common.utils as dashboard_utils
from dashboards.common.utils import select_part_dataset


def main():
    st.title("Validation Prompts Management")

    # Initialize session state for page management
    if "page" not in st.session_state:
        st.session_state.page = "Validation Prompts"

    # Sidebar for navigation
    st.sidebar.title("Navigation")
    page = st.sidebar.radio("Go to", ["Validation Prompts", "Defined Prompts", "Prompt Generator"])

    # Update the current page
    st.session_state.page = page

    # Display the selected page
    if st.session_state.page == "Validation Prompts":
        show_validation_prompts()
    elif st.session_state.page == "Defined Prompts":
        show_defined_prompts()
    elif st.session_state.page == "Prompt Generator":
        show_prompt_generator()


def show_validation_prompts():
    st.header("Validation Prompts in dataset")
    validation_prompts = database_manager.get_all_validation_prompts()
    print(validation_prompts)
    for idx, row in validation_prompts.iterrows():

        def remove_from_validation():
            database_manager.remove_validation_prompt(validation_prompt_id=row["example_id"])

        prompt_notes = json.loads(row["prompt_notes"])
        prompt_notes_df = pd.DataFrame(prompt_notes)
        prompt_piece = ff.MidiPiece(prompt_notes_df)
        parameters = row[database_manager.parameter_dtype.keys()].to_dict()
        print(parameters)
        st.json(parameters)
        streamlit_pianoroll.from_fortepyan(prompt_piece)
        st.button("Remove from validation", on_click=remove_from_validation, key=f"remove_{idx}")


def show_defined_prompts():
    st.header("Defined prompts")
    all_prompts = database_manager.get_all_prompt_notes()
    for idx, row in all_prompts.iterrows():
        prompt_notes = json.loads(row["prompt_notes"])
        prompt_notes = pd.DataFrame(prompt_notes)
        prompt_piece = ff.MidiPiece(prompt_notes)
        streamlit_pianoroll.from_fortepyan(prompt_piece)


def show_prompt_generator():
    st.header("Prompt Generator")
    col1, col2, col3 = st.columns(3)
    with col1:
        dataset_path = st.text_input(
            "Dataset Path",
            value="roszcz/maestro-sustain-v2",
            help="Enter the path to the dataset",
        )
    with col2:
        dataset_split = st.selectbox(
            "Dataset Split",
            options=["validation", "train", "test"],
            help="Choose the dataset split to use",
        )
    with col3:
        extraction_type = st.selectbox(
            "Extraction Type",
            options=["bass"],
            help="Select the type of notes to extract",
        )

    with st.spinner("Loading dataset..."):
        dataset = dashboard_utils.load_hf_dataset(
            dataset_path=dataset_path,
            dataset_split=dataset_split,
        )
        dataset = select_part_dataset(midi_dataset=dataset)

    st.header("Generation Parameters")
    with st.form("generate_parameters"):
        col1, col2 = st.columns(2)
        with col1:
            temperature = st.number_input(
                "Temperature",
                value=1.0,
                help="Controls randomness in generation",
            )
            max_new_tokens = st.number_input(
                "Max New Tokens",
                value=1024,
                help="Maximum number of new tokens to generate",
            )
        with col2:
            prompt_context_duration = st.number_input(
                "Prompt Context Duration",
                min_value=1.0,
                max_value=30.0,
                value=10.0,
                help="Duration of the prompt context in seconds",
            )
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
            prompt_duration = st.number_input(
                "Whole prompt duration", value=10.0, help="Duration of the whole prompt for iterative generation"
            )
            prompt_creation_time_step = st.number_input(
                "Prompt Creation Time Step",
                value=120.0,
                help="Time step for creating prompts",
            )
        generation_parameters = {
            "temperature": temperature,
            "max_new_tokens": max_new_tokens,
            "prompt_context_duration": prompt_context_duration,
            "target_context_duration": target_context_duration,
            "time_step": time_step,
            "task": "bass_prediction",
        }
        st.form_submit_button("Generate Prompts")
    st.image("dashboards/img/iterative_generation.png")

    st.header("Generated prompt")
    prompts: list[dict] = []

    with st.spinner("Slicing the records into prompts"):
        for record in dataset:
            prompts += prepare_prompts(
                record=record,
                extraction_type=extraction_type,
                prompt_duration=prompt_duration,
                time_step=prompt_creation_time_step,
                target_context_duration=target_context_duration,
            )

    for idx, prompt in enumerate(prompts):
        source_notes = prompt.pop("source_notes")
        target_notes = prompt.pop("target_prompt")
        prompt["prompt_notes"] = prompt["prompt_notes"].to_dict()

        def add_to_database(prompt):
            database_manager.insert_validation_prompt(
                prompt=prompt,
                parameters=generation_parameters,
            )

        source_piece = ff.MidiPiece(source_notes)
        target_piece = ff.MidiPiece(target_notes)
        streamlit_pianoroll.from_fortepyan(piece=source_piece, secondary_piece=target_piece)
        st.button(
            "Add to validation table",
            key=f"add_{idx}",
            on_click=add_to_database,
            kwargs={"prompt": prompt},
        )


if __name__ == "__main__":
    main()
