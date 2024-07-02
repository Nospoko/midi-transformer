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

from gpt2.model import GPT
import dashboards.common.utils as dashboard_utils
from dashboards.common.components import download_button
from data.tokenizer import AwesomeTokenizer, ExponentialTokenizer
from artifacts import get_voice_range, get_source_extraction_token


def generate_bass_iteratively(
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
    prompt_pieces = []  # debugging
    it = 0
    # Iterate through the piece, generating bass notes in steps
    while time + time_step < end:
        # Calculate the start offset for the bass notes in this step
        start_offset = it * time_step
        step_prompt_notes.start -= start_offset
        step_prompt_notes.end -= start_offset

        bass_prompt = step_bass_notes
        bass_prompt_piece = ff.MidiPiece(bass_prompt)
        source_piece = ff.MidiPiece(step_prompt_notes)
        prompt_pieces.append((source_piece, bass_prompt_piece))
        st.write(step_bass_notes.start.min())
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
        st.write(input_sequence)
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
    return pd.concat(all_bass_notes), prompt_pieces


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
    with st.sidebar:
        # Select device and checkpoint path
        devices = [f"cuda:{it}" for it in range(torch.cuda.device_count())] + ["cpu"]
        device = st.selectbox("device", options=devices)
        checkpoint_path = st.selectbox("checkpoint", options=glob("checkpoints/*/*.pt"))

        torch.manual_seed(4)
        torch.cuda.manual_seed(4)
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        device_type = "cuda" if "cuda" in device else "cpu"

        checkpoint = dashboard_utils.load_checkpoint(
            checkpoint_path=checkpoint_path,
            device=device,
        )
        best_val_loss = checkpoint["best_val_loss"]
        st.write(f"Model best val loss: {best_val_loss:.4f}")
        if "wandb" in dict(checkpoint).keys():
            st.link_button(label="wandb run", url=checkpoint["wandb"])

    cfg, _, tokenizer = dashboard_utils.load_tokenizer(checkpoint)
    ptdtype = {"float32": torch.float32, "bfloat16": torch.bfloat16, "float16": torch.float16}[cfg.system.dtype]
    ctx = nullcontext() if device_type == "cpu" else torch.amp.autocast(device_type=device_type, dtype=ptdtype)

    dataset_path = st.text_input("dataset", value="roszcz/maestro-sustain-v2")
    dataset_split = st.selectbox("split", options=["validation", "train", "test"])
    extraction_type = st.selectbox("extraction type", options=["bass"])

    dataset = dashboard_utils.load_hf_dataset(
        dataset_path=dataset_path,
        dataset_split=dataset_split,
    )

    # Select part of the dataset
    dataset = dashboard_utils.select_part_dataset(midi_dataset=dataset)

    # Get the record id from user input
    idx = st.number_input("record_id", value=0, max_value=len(dataset))
    record = dataset[idx]
    source = json.loads(record["source"])
    source_notes, target_notes = prepare_record(record=record, extraction_type=extraction_type)

    st.write(f"Model input size: {cfg.data.sequence_length}")

    with st.form("generate parameters"):
        temperature = st.number_input("temperature", value=1.0)
        max_new_tokens = st.number_input("max_new_tokens", value=cfg.data.sequence_length)
        prompt_context_duration = st.number_input("prompt_context_duration", value=10.0)
        target_context_duration = st.number_input("target_contex_duration", value=0.0)
        time_step = st.number_input("time_step", value=10.0)
        run = st.form_submit_button("Generate")

    if not run:
        return

    # Decode and display the original piece
    notes = pd.concat([source_notes, target_notes], ignore_index=True)
    notes = notes.sort_values(by="start").reset_index(drop=True)

    bass_prompt = target_notes[target_notes.end < target_context_duration]
    bass_prompt_piece = ff.MidiPiece(bass_prompt)

    source_piece = ff.MidiPiece(source_notes)
    target_piece = ff.MidiPiece(target_notes)

    if source_piece.size == 0 and bass_prompt_piece.size == 0:
        st.write("Warning: Empty prompt!")
    else:
        st.write("Prompt piece")
        streamlit_pianoroll.from_fortepyan(piece=source_piece, secondary_piece=bass_prompt_piece)

    piece = ff.MidiPiece(notes, source=source)

    pad_token_id = tokenizer.token_to_id["<PAD>"]
    model = dashboard_utils.initialize_model(
        cfg,
        checkpoint=checkpoint,
        device=device,
        pad_token_id=pad_token_id,
    )

    # Generate new tokens and create the generated piece
    prefix_token = get_source_extraction_token(extraction_type=extraction_type)
    note_token_ids = tokenizer.encode(
        source_notes,
        prefix_tokens=[prefix_token],
    )
    bass_token_id = tokenizer.token_to_id["<BASS>"]
    note_token_ids.append(bass_token_id)
    st.write(f"Input sequence tokens size: {len(note_token_ids)}")

    with ctx:
        bass_notes, prompt_pieces = generate_bass_iteratively(
            model=model,
            tokenizer=tokenizer,
            prompt_notes=source_notes,
            target_notes=target_notes,
            prompt_context_duration=prompt_context_duration,
            target_context_duration=target_context_duration,
            time_step=time_step,
            device=device,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
        )

    st.write("Prompt pieces")
    for prompt_piece, bass_prompt_piece in prompt_pieces:
        try:
            streamlit_pianoroll.from_fortepyan(piece=prompt_piece, secondary_piece=bass_prompt_piece)
        except DuplicateWidgetID:
            st.write("Duplicate pianoroll")
            pass

    st.write(bass_notes)
    bass_piece = ff.MidiPiece(bass_notes)

    io_columns = st.columns(2)
    title, composer = source["title"], source["composer"]
    piece_name = (title + composer).replace(" ", "_").casefold()

    # Display and allow download of the original MIDI
    with io_columns[0]:
        st.write("original:")
        streamlit_pianoroll.from_fortepyan(piece=source_piece, secondary_piece=target_piece)
        original_midi_path = f"tmp/fragment_of_{piece_name}_{idx}.mid"
        source_file = piece.to_midi()
        source_file.write(original_midi_path)
        with open(original_midi_path, "rb") as file:
            st.markdown(
                download_button(file.read(), original_midi_path.split("/")[-1], "Download source midi"),
                unsafe_allow_html=True,
            )
        os.unlink(original_midi_path)

    # Display and allow download of the generated MIDI
    with io_columns[1]:
        st.write("generated:")
        streamlit_pianoroll.from_fortepyan(piece=bass_piece)
        milion_parameters = model.get_num_params() / 1e6
        midi_path = f"tmp/{milion_parameters:.0f}_variations_on_{piece_name}_{idx}.mid"
        generated_file = bass_piece.to_midi()
        generated_file.write(midi_path)
        with open(midi_path, "rb") as file:
            st.markdown(
                download_button(file.read(), midi_path.split("/")[-1], "Download generated midi"),
                unsafe_allow_html=True,
            )
        os.unlink(midi_path)

        # with st.expander("Tokens"):
        #     st.write(tokenizer.vocab[token_id] for token_id in output)

    st.write("whole")

    out_notes = pd.concat([source_notes, bass_notes]).sort_values(by="start").reindex()
    out_piece = ff.MidiPiece(out_notes)
    streamlit_pianoroll.from_fortepyan(piece=source_piece, secondary_piece=bass_piece)

    # Allow download of the full MIDI with context
    full_midi_path = f"tmp/full_{milion_parameters}_variations_on_{piece_name}_{idx}.mid"
    out_piece.to_midi().write(full_midi_path)
    with open(full_midi_path, "rb") as file:
        st.markdown(
            download_button(file.read(), full_midi_path.split("/")[-1], "Download midi with context"),
            unsafe_allow_html=True,
        )
    os.unlink(full_midi_path)


if __name__ == "__main__":
    main()
