import os
import json
from glob import glob
from contextlib import nullcontext

import torch
import pandas as pd
import fortepyan as ff
import streamlit as st
import streamlit_pianoroll

import dashboards.common.utils as dashboard_utils
from dashboards.common.components import download_button


def prepare_record(record: dict, start: float, end: float):
    notes = pd.DataFrame(record["notes"])
    notes = notes[(notes.start > start) & (notes.start < end)]

    notes.end -= notes.start.min()
    notes.start -= notes.start.min()
    return notes


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

    # Decode and display the original piece
    start = st.number_input(label="start second", value=0.0)
    end = st.number_input(label="end second", value=60.0)
    notes = prepare_record(record=record, start=start, end=end)

    notes_pressed_at_end = notes[(notes.start < end) & (notes.end > end)]
    n_notes_pressed = len(notes_pressed_at_end)

    piece = ff.MidiPiece(notes, source=source)

    with st.form("generate parameters"):
        temperature = st.number_input("temperature", value=1.0)
        max_new_tokens = st.number_input("max_new_tokens", value=cfg.data.sequence_length)
        run = st.form_submit_button("Generate")

    if not run:
        return

    pad_token_id = tokenizer.token_to_id["<PAD>"]
    model = dashboard_utils.initialize_model(cfg, checkpoint=checkpoint, device=device, pad_token_id=pad_token_id)
    # Generate new tokens and create the generated piece - generation does not require padding
    note_token_ids = tokenizer.encode(
        notes=notes,
    )
    # Cut the notes to end the sequence roughly at the exact timestamp.
    # This should help with model performance and generate better sounding sequences
    if n_notes_pressed > 0:
        note_token_ids = note_token_ids[: -3 * n_notes_pressed]
    input_sequence = torch.tensor([note_token_ids], device=device)
    with torch.no_grad():
        with ctx:
            output = model.generate(input_sequence, max_new_tokens=max_new_tokens, temperature=temperature)

    output = output[0].cpu().numpy()
    out_notes = tokenizer.decode(output)
    out_piece = ff.MidiPiece(out_notes)
    generated_notes = out_notes.iloc[piece.size :].copy()
    generated_piece = ff.MidiPiece(generated_notes)
    generated_piece.time_shift(-generated_piece.df.start.min())

    io_columns = st.columns(2)
    title, composer = source["title"], source["composer"]
    piece_name = (title + composer).replace(" ", "_").casefold()

    # Display and allow download of the original MIDI
    with io_columns[0]:
        st.write("original:")
        streamlit_pianoroll.from_fortepyan(piece=piece)
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
        streamlit_pianoroll.from_fortepyan(piece=generated_piece)
        output = output[input_sequence.shape[-1] :]
        milion_parameters = model.get_num_params() / 1e6
        midi_path = f"tmp/{milion_parameters:.0f}_variations_on_{piece_name}_{idx}.mid"
        generated_file = generated_piece.to_midi()
        generated_file.write(midi_path)
        with open(midi_path, "rb") as file:
            st.markdown(
                download_button(file.read(), midi_path.split("/")[-1], "Download generated midi"),
                unsafe_allow_html=True,
            )
        os.unlink(midi_path)

        with st.expander("Tokens"):
            st.write(f"{tokenizer.vocab[token_id]} " for token_id in output)

    st.write("whole model output")
    second_part = ff.MidiPiece(out_notes[piece.size :].copy())
    expanded_piece = ff.MidiPiece(out_notes[: piece.size].copy())
    streamlit_pianoroll.from_fortepyan(piece=expanded_piece, secondary_piece=second_part)

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
