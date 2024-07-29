import json

import pandas as pd
import fortepyan as ff
import streamlit as st
import streamlit_pianoroll
from streamlit.errors import DuplicateWidgetID

import gpt2.generation as generation
import data.database_manager as database_manager
import dashboards.common.utils as dashboard_utils
from dashboards.common.utils import select_part_dataset, select_generation_parameters


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
    task_options = ["bass_prediction", "reverse_bass_prediction", "next_token_prediction"]
    task = st.selectbox("task", options=task_options)
    validation_prompts = database_manager.get_validation_examples_for_task(task=task)
    for idx, row in validation_prompts.iterrows():

        def remove_from_validation(row):
            database_manager.remove_validation_prompt(validation_prompt_id=row["example_id"])

        prompt_notes = json.loads(row["prompt_notes"])
        prompt_notes_df = pd.DataFrame(prompt_notes)
        prompt_piece = ff.MidiPiece(prompt_notes_df)

        parameters = row[database_manager.parameter_dtype.keys()].to_dict()
        prompt = row[database_manager.prompt_dtype.keys()].to_dict()
        prompt.pop("prompt_notes")

        json_columns = st.columns(2)
        json_columns[0].json(parameters)
        json_columns[1].json(prompt)

        try:
            streamlit_pianoroll.from_fortepyan(piece=prompt_piece)
        except DuplicateWidgetID:
            st.write("Duplicate widget")
        st.button(
            "Remove from validation",
            on_click=remove_from_validation,
            key=f"remove_{idx}",
            kwargs={"row": row},
        )


def show_defined_prompts():
    st.header("Defined prompts")
    all_prompts = database_manager.get_all_prompt_notes()
    for idx, row in all_prompts.iterrows():
        prompt = row[database_manager.prompt_dtype.keys()].to_dict()
        prompt.pop("prompt_notes")

        st.json(prompt)
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
        prediction_task = st.selectbox(
            "Extraction Type",
            options=["bass_prediction", "reverse_bass_prediction", "-"],
            help="Select the type of notes to extract",
        )

    with st.spinner("Loading dataset..."):
        dataset = dashboard_utils.load_hf_dataset(
            dataset_path=dataset_path,
            dataset_split=dataset_split,
        )
        dataset = select_part_dataset(midi_dataset=dataset)

    _, generation_parameters, prompt_duration, prompt_creation_time_step = select_generation_parameters()
    task = generation_parameters["task"]

    st.header("Generated prompt")
    prompts: list[dict] = []

    with st.spinner("Slicing the records into prompts"):
        for record in dataset:
            if task == "next_token_prediction":
                prompts += generation.prepare_next_token_prediction_prompts(
                    record=record,
                    prompt_duration=prompt_duration,
                    time_step=prompt_creation_time_step,
                )
            else:
                prompts += generation.prepare_subsequence_prediction_prompts(
                    record=record,
                    prediction_task=prediction_task,
                    prompt_duration=prompt_duration,
                    time_step=prompt_creation_time_step,
                    target_context_duration=generation_parameters["target_context_duration"],
                )

    for idx, prompt in enumerate(prompts):
        if task == "next_token_prediction":
            source_notes = prompt["prompt_notes"]

            source_piece = ff.MidiPiece(source_notes)
            try:
                streamlit_pianoroll.from_fortepyan(piece=source_piece)
            except DuplicateWidgetID:
                st.write("Duplicate widget")
        else:
            source_notes = prompt.pop("source_notes")
            target_notes = prompt.pop("target_prompt")

            source_piece = ff.MidiPiece(source_notes)
            target_piece = ff.MidiPiece(target_notes)
            try:
                streamlit_pianoroll.from_fortepyan(piece=source_piece, secondary_piece=target_piece)
            except DuplicateWidgetID:
                st.write("Duplicate widget")

        prompt["prompt_notes"] = prompt["prompt_notes"].to_dict()

        def add_to_database(prompt):
            database_manager.insert_validation_prompt(
                prompt=prompt,
                parameters=generation_parameters,
            )

        st.button(
            "Add to validation table",
            key=f"add_{idx}",
            on_click=add_to_database,
            kwargs={"prompt": prompt},
        )


if __name__ == "__main__":
    main()
