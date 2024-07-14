import os
import json
from glob import glob
from contextlib import nullcontext

import torch
import pandas as pd
import fortepyan as ff
import streamlit as st
import streamlit_pianoroll
from streamlit.errors import DuplicateWidgetID

from artifacts import get_voice_range
import dashboards.common.utils as dashboard_utils
from dashboards.common.components import download_button
from gpt2.generation import generate_subsequence_iteratively
from gpt2.utils import load_cfg, load_tokenizer, initialize_model


def prepare_record(record: dict, extraction_type: str):
    """
    Prepare a record for note extraction based on the specified type.

    Args:
        record: Dictionary containing note data
        extraction_type: Type of extraction (e.g., 'bass')

    Returns:
        Tuple of DataFrames (source_notes, target_notes)
    """
    low, high = get_voice_range(voice=extraction_type)
    start_end_columns = st.columns(2)
    start = start_end_columns[0].number_input(label="start second", value=0.0)
    end = start_end_columns[1].number_input(label="end second", value=60.0)

    notes = pd.DataFrame(record["notes"])
    notes = notes[(notes.start > start) & (notes.end < end)]
    notes.end -= notes.start.min()
    notes.start -= notes.start.min()
    extracted_ids = (notes.pitch >= low) & (notes.pitch < high)
    source_notes = notes[~extracted_ids]
    target_notes = notes[extracted_ids]

    return source_notes, target_notes


def main():
    st.title("🎵 Bass Generation Dashboard")
    st.markdown(
        """
    This dashboard allows you to generate bass lines for musical pieces using a GPT model.
    Select your parameters, choose a piece, and let AI compose for you!
    """
    )

    with st.sidebar:
        st.header("Model Configuration")
        devices = [f"cuda:{it}" for it in range(torch.cuda.device_count())] + ["cpu"]
        device = st.selectbox("Select Device", options=devices, help="Choose the device to run the model on")
        checkpoint_path = st.selectbox(
            "Select Checkpoint",
            options=glob("checkpoints/*.pt"),
            help="Choose the model checkpoint to use",
        )

        with st.spinner("Loading checkpoint..."):
            checkpoint = torch.load(
                f=checkpoint_path,
                map_location=device,
            )

        st.success(f"Model loaded! Best validation loss: {checkpoint['best_val_loss']:.4f}")
        if "wandb" in dict(checkpoint).keys():
            st.link_button(label="View Training Run", url=checkpoint["wandb"])

    cfg = load_cfg(checkpoint=checkpoint)
    tokenizer = load_tokenizer(cfg=cfg)
    ptdtype = {"float32": torch.float32, "bfloat16": torch.bfloat16, "float16": torch.float16}[cfg.system.dtype]
    device_type = "cuda" if "cuda" in device else "cpu"
    ctx = (
        nullcontext()
        if device_type == "cpu"
        else torch.amp.autocast(
            device_type=device_type,
            dtype=ptdtype,
        )
    )

    tab1, tab2, tab3 = st.tabs(["Dataset Selection", "Generation Parameters", "Results"])

    with tab1:
        st.header("Dataset Configuration")
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
            dataset = dashboard_utils.select_part_dataset(midi_dataset=dataset)

        st.success(f"Dataset loaded! Total records: {len(dataset)}")

        idx = st.number_input(
            "Select Record ID",
            value=0,
            max_value=len(dataset) - 1,
            help="Choose a specific record from the dataset",
        )
        record = dataset[idx]
        source = json.loads(record["source"])
        st.info(f"Selected piece: '{source['title']}' by {source['composer']}")

    with tab2:
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
                    value=cfg.data.sequence_length,
                    help="Maximum number of new tokens to generate",
                )
            with col2:
                prompt_context_duration = st.slider(
                    "Prompt Context Duration",
                    min_value=1.0,
                    max_value=30.0,
                    value=10.0,
                    help="Duration of the prompt context in seconds",
                )
                target_context_duration = st.slider(
                    "Target Context Duration",
                    min_value=0.0,
                    max_value=30.0,
                    value=0.0,
                    help="Duration of the bass context in seconds",
                )
                time_step = st.slider(
                    "Time Step",
                    min_value=1.0,
                    max_value=30.0,
                    value=10.0,
                    help="Time step for generation in seconds",
                )

            run = st.form_submit_button("Generate Bass Line")
        st.image("dashboards/img/iterative_generation.png")

    if run:
        with tab3:
            st.header("Generation Results")
            with st.spinner("Preparing data..."):
                source_notes, target_notes = prepare_record(record=record, extraction_type=extraction_type)
                notes = pd.concat([source_notes, target_notes], ignore_index=True)
                notes = notes.sort_values(by="start").reset_index(drop=True)
                bass_prompt = target_notes[target_notes.end < target_context_duration].copy()

                source_piece = ff.MidiPiece(source_notes)
                bass_prompt_piece = ff.MidiPiece(bass_prompt)

            if source_piece.size == 0 and bass_prompt_piece.size == 0:
                st.warning("Warning: Empty prompt! Generation may not produce meaningful results.")

            st.subheader("Original Piece with Bass Prompt")
            streamlit_pianoroll.from_fortepyan(piece=source_piece, secondary_piece=bass_prompt_piece)

            with st.spinner("Generating bass line..."):
                pad_token_id = tokenizer.token_to_id["<PAD>"]
                model = initialize_model(
                    cfg,
                    checkpoint=checkpoint,
                    device=device,
                    pad_token_id=pad_token_id,
                )

                bass_notes = generate_subsequence_iteratively(
                    model=model,
                    tokenizer=tokenizer,
                    prompt_notes=source_notes,
                    target_notes=target_notes,
                    prompt_context_duration=prompt_context_duration,
                    target_context_duration=target_context_duration,
                    time_step=time_step,
                    device=device,
                    ctx=ctx,
                    max_new_tokens=max_new_tokens,
                    temperature=temperature,
                )

            st.success("Bass line generated successfully!")

            st.subheader("Generated Bass Line")
            bass_piece = ff.MidiPiece(bass_notes)
            streamlit_pianoroll.from_fortepyan(piece=bass_piece)

            st.subheader("Combined Result")
            out_notes = pd.concat([source_notes, bass_notes]).sort_values(by="start").reset_index(drop=True)
            out_piece = ff.MidiPiece(out_notes)

            try:
                streamlit_pianoroll.from_fortepyan(piece=source_piece, secondary_piece=bass_piece)
            except DuplicateWidgetID:
                st.write("Duplicate pianoroll")

            # Download buttons
            col1, col2, col3 = st.columns(3)
            with col1:
                download_midi(
                    source_piece,
                    f"original_{source['title']}.mid",
                    "Download Original MIDI",
                )
            with col2:
                download_midi(
                    bass_piece,
                    f"generated_bass_{source['title']}.mid",
                    "Download Generated Bass MIDI",
                )
            with col3:
                download_midi(
                    out_piece,
                    f"combined_{source['title']}.mid",
                    "Download Combined MIDI",
                )


def download_midi(piece, filename, button_text):
    piece.to_midi().write(filename)
    with open(filename, "rb") as file:
        st.markdown(
            download_button(file.read(), filename.split("/")[-1], button_text),
            unsafe_allow_html=True,
        )
    os.unlink(filename)


if __name__ == "__main__":
    main()
