import os
from glob import glob
from contextlib import nullcontext

import yaml
import torch
import streamlit as st
from datasets import Dataset

import data.database_manager as database_manager
import dashboards.common.utils as dashboard_utils
from artifacts import get_source_extraction_token
from gpt2.generation import prepare_prompts, generate_bass_iteratively


def multiselect_part_dataset(midi_dataset: Dataset) -> Dataset:
    """
    Allows the user to select a part of the dataset based on composers and titles.

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
    selected_composers = st.multiselect("Select composers", options=composers, default=composers[:3])

    ids = source_df.composer.isin(selected_composers)
    piece_titles = source_df[ids].title.unique()
    selected_titles = st.multiselect("Select titles", options=piece_titles, default=piece_titles[:3])

    ids = source_df.composer.isin(selected_composers) & source_df.title.isin(selected_titles)
    part_df = source_df[ids]
    part_dataset = midi_dataset.select(part_df.index.values)

    return part_dataset


def main():
    st.title("🎵 Bass Generation Dashboard for populating the database")

    with st.sidebar:
        st.header("Model Configuration")
        devices = [f"cuda:{it}" for it in range(torch.cuda.device_count())] + ["cpu"]
        device = st.selectbox("Select Device", options=devices, help="Choose the device to run the model on")
        checkpoint_path = st.selectbox(
            "Select Checkpoint",
            options=glob("checkpoints/*/*.pt"),
            help="Choose the model checkpoint to use",
        )

        with st.spinner("Loading checkpoint..."):
            checkpoint = dashboard_utils.load_checkpoint(
                checkpoint_path=checkpoint_path,
                device=device,
            )

        run_name = os.path.basename(checkpoint_path)
        model_registration, _ = database_manager.register_model_from_checkpoint(
            checkpoint=checkpoint,
            run_name=run_name,
        )

        st.success(f"Model loaded! Best validation loss: {checkpoint['best_val_loss']:.4f}")
        if "wandb" in dict(checkpoint).keys():
            st.link_button(label="View Training Run", url=checkpoint["wandb"])

    cfg, _, tokenizer = dashboard_utils.load_tokenizer(checkpoint)
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
            dataset = multiselect_part_dataset(midi_dataset=dataset)

        st.success(f"Dataset loaded! Total records: {len(dataset)}")

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
                    value=10.0,
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
            run = st.form_submit_button("Generate Bass Line")
        st.image("dashboards/img/iterative_generation.png")

    if run:
        with tab3:
            st.header("Generation state")
            pad_token_id = tokenizer.token_to_id["<PAD>"]

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

            model = dashboard_utils.initialize_model(
                cfg,
                checkpoint=checkpoint,
                device=device,
                pad_token_id=pad_token_id,
            )
            num_prompts = len(prompts)

            for idx, prompt in enumerate(prompts):
                source_notes = prompt.pop("source_notes")
                bass_prompt = prompt.pop("target_prompt")

                with st.spinner(f"Generating bass line... {idx} / {num_prompts}"):
                    prefix_token = get_source_extraction_token(extraction_type=extraction_type)
                    note_token_ids = tokenizer.encode(source_notes, prefix_tokens=[prefix_token])
                    note_token_ids.append(tokenizer.token_to_id["<BASS>"])

                    with ctx:
                        bass_notes = generate_bass_iteratively(
                            model=model,
                            tokenizer=tokenizer,
                            prompt_notes=source_notes,
                            target_notes=bass_prompt,
                            prompt_context_duration=prompt_context_duration,
                            target_context_duration=target_context_duration,
                            time_step=time_step,
                            device=device,
                            max_new_tokens=max_new_tokens,
                            temperature=temperature,
                        )

                generated_notes = bass_notes.iloc[len(bass_prompt) :]
                database_manager.insert_generated_notes(
                    model=model_registration,
                    prompt=prompt,
                    parameters=generation_parameters,
                    generated_notes=generated_notes,
                )
            st.success(f"Prepared {num_prompts} generations")


if __name__ == "__main__":
    main()
