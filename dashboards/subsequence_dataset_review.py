import torch
import fortepyan as ff
import streamlit as st
import streamlit_pianoroll
import matplotlib.pyplot as plt
from datasets import load_dataset

from data.tokenizer import ExponentialTokenizer
from data.subsequence_dataset import SubSequenceMidiDataset
from artifacts import special_tokens, get_target_extraction_token


def plot_target_mask(target_mask):
    fig, ax = plt.subplots(figsize=(12, 2))
    ax.imshow(target_mask.unsqueeze(0), cmap="binary", aspect="auto")
    ax.set_yticks([])
    ax.set_xlabel("Token Position")
    ax.set_title("Target Mask")
    return fig


def main():
    st.title("MIDI Dataset Review Dashboard")

    dataset_names = [
        "BassExtractedDataset",
        # Add more dataset names here
    ]
    dataset_name = st.selectbox(label="Dataset", options=dataset_names)
    dataset_split = st.selectbox(label="Split", options=["train", "test", "validation"])

    with st.form(key="config_form"):
        col1, col2 = st.columns(2)
        with col1:
            base_dataset_name = st.text_input(label="Base Dataset Name", value="roszcz/maestro-sustain-v2")
            extra_datasets = st.text_input(label="Extra Datasets (comma separated)", value="")
            notes_per_record = st.number_input(label="Notes per Record", min_value=1, value=512)
            step = st.number_input(label="Step", min_value=1, value=512)
        with col2:
            pause_detection_threshold = st.number_input(label="Pause Detection Threshold", value=4.0)
            sequence_length = st.number_input(label="Sequence Length", min_value=1, value=5000, step=500)
            loss_calculation_style = st.selectbox(label="Loss Calculation Style", options=["pretraining", "finetuning"])

        st.form_submit_button(label="Update Config")

    with st.form(key="tokenizer_form"):
        col1, col2 = st.columns(2)
        with col1:
            min_time_unit = st.number_input(label="Min Time Unit", min_value=0.01, value=0.01, step=0.01, format="%.2f")
        with col2:
            n_velocity_bins = st.number_input(label="Velocity Bins", min_value=1, value=32, step=1)

        st.form_submit_button(label="Update Tokenizer")

    extra_datasets_list = [x.strip() for x in extra_datasets.split(",") if x.strip()]

    config = {
        "base_dataset_name": base_dataset_name,
        "extra_datasets": extra_datasets_list,
        "notes_per_record": notes_per_record,
        "step": step,
        "pause_detection_threshold": pause_detection_threshold,
    }

    tokenizer_parameters = {
        "min_time_unit": min_time_unit,
        "n_velocity_bins": n_velocity_bins,
        "special_tokens": special_tokens,
    }

    tokenizer = ExponentialTokenizer(**tokenizer_parameters)

    dataset = load_dataset(
        f"midi_datasets/{dataset_name}",
        split=dataset_split,
        trust_remote_code=True,
        num_proc=8,
        **config,
    )
    midi_dataset = SubSequenceMidiDataset(
        dataset=dataset,
        tokenizer=tokenizer,
        sequence_length=sequence_length,
        loss_calculation_style=loss_calculation_style,
    )

    total_tokens = config["notes_per_record"] * dataset.num_rows
    st.write(f"Total Rows: {dataset.num_rows}")
    st.write(f"Total Notes: {total_tokens}")

    with st.expander("Configuration"):
        st.json(config)

    with st.expander("Tokenizer Parameters"):
        st.json(tokenizer_parameters)

    idx = st.number_input(label="Record ID", value=0, min_value=0, max_value=len(dataset) - 1)
    record = midi_dataset[idx]

    with st.expander(label="Source Data"):
        st.json(record["source"])

    extracted = record["extraction_type"]
    st.write(f"Extraction Type: {extracted}")

    src_token_ids = record["source_token_ids"]
    tgt_token_ids = record["target_token_ids"]

    st.write("### Token Information")
    col1, col2 = st.columns(2)
    with col1:
        st.write("Source Tokens:")
        st.write(src_token_ids)
    with col2:
        st.write("Target Tokens:")
        st.write(tgt_token_ids)

    st.write("### Target Mask Visualization")
    target_mask_fig = plot_target_mask(record["target_mask"])
    st.pyplot(target_mask_fig)

    # Display some statistics about the target mask
    true_count = torch.sum(record["target_mask"]).item()
    total_count = len(record["target_mask"])
    true_percentage = (true_count / total_count) * 100

    st.write("Target Mask Statistics:")
    st.write(f"- Total tokens: {total_count}")
    st.write(f"- Tokens used for loss calculation: {true_count}")
    st.write(f"- Percentage of tokens used: {true_percentage:.2f}%")

    src_tokens = [midi_dataset.tokenizer.vocab[token_id] for token_id in src_token_ids]

    extraction_token = get_target_extraction_token(extracted)
    extraction_position = src_tokens.index(extraction_token)
    prompt_tokens = src_tokens[:extraction_position]
    extracted_tokens = src_tokens[extraction_position:]

    prompt_notes = midi_dataset.tokenizer.untokenize(prompt_tokens)
    extracted_notes = midi_dataset.tokenizer.untokenize(extracted_tokens)

    src_piece = ff.MidiPiece(prompt_notes)
    tgt_piece = ff.MidiPiece(extracted_notes)

    st.write("### Visualizations")
    st.write("#### Combined View:")
    streamlit_pianoroll.from_fortepyan(piece=src_piece, secondary_piece=tgt_piece)

    col1, col2 = st.columns(2)
    with col1:
        st.write("#### Prompt:")
        streamlit_pianoroll.from_fortepyan(piece=src_piece)
    with col2:
        st.write("#### Extracted:")
        streamlit_pianoroll.from_fortepyan(piece=tgt_piece)


if __name__ == "__main__":
    main()
