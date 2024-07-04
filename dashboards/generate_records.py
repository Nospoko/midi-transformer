import os
import json
from glob import glob
from contextlib import nullcontext

import yaml
import torch
import pandas as pd
import streamlit as st
from datasets import Dataset

from gpt2.model import GPT
import dashboards.common.utils as dashboard_utils
import dashboards.common.database_manager as database_manager
from data.tokenizer import AwesomeTokenizer, ExponentialTokenizer
from artifacts import get_voice_range, get_source_extraction_token


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


def generate_bass(
    model: GPT,
    tokenizer: ExponentialTokenizer | AwesomeTokenizer,
    prompt_notes: pd.DataFrame,
    target_notes: pd.DataFrame,
    prompt_context_duration: float,
    target_context_duration: float,
    time_step: float,
    device: torch.device,
    temperature: float = 1.0,
    max_new_tokens: int = 512,
) -> pd.DataFrame:
    """
    Generate bass notes iteratively using the given model and tokenizer.

    Args:
        model: The GPT model for generation
        tokenizer: The tokenizer for encoding/decoding notes
        prompt_notes: DataFrame containing prompt notes
        target_notes: DataFrame containing target notes
        prompt_context_duration: Duration of the prompt context
        target_context_duration: Duration of the target context
        device: The device to run the model on
        temperature: Temperature for sampling
        max_new_tokens: Maximum number of new tokens to generate

    Returns:
        DataFrame containing generated bass notes
    """
    # Initialize the first step with notes within the prompt and target context durations
    step_prompt_notes = prompt_notes[prompt_notes.end < prompt_context_duration]
    step_bass_notes = target_notes[target_notes.end < target_context_duration]
    # Initialize the list of all bass notes with the initial target notes
    all_bass_notes = [step_bass_notes]
    time = 0
    end = prompt_notes.end.max()

    # Handle the case where there's no target context
    if target_context_duration == 0:
        step_bass_notes = pd.DataFrame(columns=prompt_notes.columns)
    it = 0
    # Iterate through the piece, generating bass notes in steps
    while time + time_step < end:
        # Calculate the start offset for the bass notes in this step
        start_offset = it * time_step
        it += 1
        step_prompt_notes.start -= start_offset
        step_prompt_notes.end -= start_offset

        step_bass_notes = step_bass_notes[(step_bass_notes.start > 0) & (step_bass_notes.end > 0)]
        # Tokenize the current step's prompt and target notes
        step_sequence = tokenizer.tokenize(step_prompt_notes)
        step_bass = tokenizer.tokenize(step_bass_notes)

        # Combine prompt, bass marker, and target into input sequence
        input_sequence = step_sequence + ["<BASS>"] + step_bass
        # Convert tokens to ids and prepare input tensor
        input_token_ids = torch.tensor(
            [[tokenizer.token_to_id[token] for token in input_sequence]],
            device=device,
        )
        print(f"generating {time} - {time + prompt_context_duration} with {target_context_duration} target context")
        # Generate new tokens using the model
        output = model.generate(
            idx=input_token_ids,
            temperature=temperature,
            max_new_tokens=max_new_tokens,
        )
        print("generation successful")
        # Convert output to numpy array and decode tokens
        output = output[0].cpu().numpy()
        out_tokens = [tokenizer.vocab[token_id] for token_id in output]

        # Extract bass tokens (everything after the <BASS> marker)
        bass_command_position = out_tokens.index("<BASS>")
        bass_tokens = out_tokens[bass_command_position:].copy()

        # Convert bass tokens back to notes
        output_bass_notes = tokenizer.untokenize(bass_tokens)

        # Select only the newly generated notes within the current time step
        notes_after_context = output_bass_notes.start > target_context_duration
        notes_within_step = output_bass_notes.end < target_context_duration + time_step
        valid_new_notes = notes_after_context & notes_within_step
        bass_notes = output_bass_notes[valid_new_notes].copy()
        step_bass_notes = bass_notes.copy()

        # Adjust the start and end times of the bass notes
        bass_notes.start += start_offset
        bass_notes.end += start_offset
        bass_notes["duration"] = bass_notes.end - bass_notes.start

        # Add the generated bass notes to the collection
        all_bass_notes.append(bass_notes)
        # Prepare for the next iteration:
        # Select the prompt notes for the next time step
        time = time + time_step
        prompt_selector = (prompt_notes.start > time) & (prompt_notes.end < time + prompt_context_duration)
        step_prompt_notes = prompt_notes[prompt_selector].copy()
        step_bass_notes = step_bass_notes[step_bass_notes.start > target_context_duration + time_step]
        step_bass_notes.start -= target_context_duration + time_step
        step_bass_notes.end -= target_context_duration + time_step

    # Combine all generated bass notes and return
    return pd.concat(all_bass_notes)


def prepare_prompts(
    record: dict,
    extraction_type: str,
    time_step: float,
    prompt_context_duration: float,
    target_context_duration: float,
) -> list[dict]:
    low, high = get_voice_range(voice=extraction_type)
    time = 0

    notes = pd.DataFrame(record["notes"])
    source = json.loads(record["source"])
    prompts = []

    while time + prompt_context_duration < notes.end.max():
        start = time
        end = time + prompt_context_duration

        notes = notes[(notes.start > start) & (notes.end < end)]
        notes.end -= notes.start.min()
        notes.start -= notes.start.min()
        extracted_ids = (notes.pitch >= low) & (notes.pitch < high)
        source_notes = notes[~extracted_ids]
        target_notes = notes[extracted_ids]
        target_prompt = target_notes[target_notes.end < target_context_duration]
        prompt = {
            "source_notes": source_notes,
            "target_prompt": target_prompt,
            "prompt_notes": pd.concat([source_notes, target_prompt]),
            "start_time": notes.start.min(),
            "end_time": notes.start.max(),
            "midi_filename": source["midi_filename"],
        }
        prompts.append(prompt)
        time += time_step

    return prompts


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

        # Hard-coded for the specific naming style
        milion_parameters = run_name.split("-")[3][-1]

        model_descriptor = {
            "name": os.path.basename(checkpoint_path),
            "milion_parameters": milion_parameters,
            "best_val_loss": checkpoint["best_val_loss"],
        }

        st.success(f"Model loaded! Best validation loss: {checkpoint['best_val_loss']:.4f}")
        if "wandb" in dict(checkpoint).keys():
            st.link_button(label="View Training Run", url=checkpoint["wandb"])
            model_descriptor |= {"wandb": checkpoint["wandb"]}

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
                    "Time Step",
                    min_value=1.0,
                    max_value=30.0,
                    value=10.0,
                    help="Time step for generation in seconds",
                )
            generation_parameters = {
                "tempperature": temperature,
                "max_new_tokens": max_new_tokens,
                "prompt_context_duration": prompt_context_duration,
                "target_context_duration": target_context_duration,
                "task": "bass_prediction",
            }
            run = st.form_submit_button("Generate Bass Line")
        st.image("dashboards/img/iterative_generation.png")

    if run:
        with tab3:
            st.header("Generation state")
            prompts: list[dict] = []
            for record in dataset:
                prompts += prepare_prompts(
                    record=record,
                    extraction_type=extraction_type,
                    prompt_context_duration=prompt_context_duration,
                    time_step=time_step,
                    target_context_duration=target_context_duration,
                )
            num_prompts = len(prompts)

            for idx, prompt in enumerate(prompts):
                source_notes = prompt.pop("source_notes")
                bass_prompt = prompt.pop("target_prompt")

                with st.spinner(f"Generating bass line... {idx} / {num_prompts}"):
                    pad_token_id = tokenizer.token_to_id["<PAD>"]
                    model = dashboard_utils.initialize_model(
                        cfg,
                        checkpoint=checkpoint,
                        device=device,
                        pad_token_id=pad_token_id,
                    )

                    prefix_token = get_source_extraction_token(extraction_type=extraction_type)
                    note_token_ids = tokenizer.encode(source_notes, prefix_tokens=[prefix_token])
                    note_token_ids.append(tokenizer.token_to_id["<BASS>"])

                    with ctx:
                        bass_notes = generate_bass(
                            model=model,
                            tokenizer=tokenizer,
                            prompt_notes=source_notes,
                            prompt_bass=bass_prompt,
                            device=device,
                            max_new_tokens=max_new_tokens,
                            temperature=temperature,
                        )

                st.success("Bass line generated successfully!")

            generated_notes = bass_notes.iloc[len(bass_prompt) :]
            database_manager.insert_generated_notes(
                model=model_descriptor,
                prompt=prompt,
                parameters=generation_parameters,
                generated_notes=generated_notes,
            )


if __name__ == "__main__":
    main()
