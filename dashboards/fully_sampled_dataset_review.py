import torch
import fortepyan as ff
import streamlit as st
import streamlit_pianoroll
import matplotlib.pyplot as plt
from datasets import load_dataset

from artifacts import special_tokens
from data.median_dataset import MedianDataset
from data.tokenizer import ExponentialTokenizer


@st.cache_data()
def load_hf_dataset(
    config: dict,
    dataset_name: str,
    dataset_split: str,
):
    dataset = load_dataset(
        f"midi_datasets/{dataset_name}",
        split=dataset_split,
        trust_remote_code=True,
        num_proc=8,
        **config,
    )
    return dataset


def plot_target_mask(target_mask):
    fig, ax = plt.subplots(figsize=(12, 2))
    ax.imshow(target_mask.unsqueeze(0), cmap="binary", aspect="auto")
    ax.set_yticks([])
    ax.set_xlabel("Token Position")
    ax.set_title("Target Mask")
    return fig


def main():
    st.title("Median MIDI Dataset Review Dashboard")

    dataset_split = st.selectbox(label="Split", options=["train", "test", "validation"])

    with st.form(key="config_form"):
        col1, col2 = st.columns(2)
        with col1:
            base_dataset_name = st.text_input(
                label="Base Dataset Name",
                value="roszcz/maestro-sustain-v2",
            )
            notes_per_record = st.number_input(
                label="Notes per Record",
                min_value=1,
                value=128,
            )
        with col2:
            sequence_length = st.number_input(label="Sequence Length", min_value=1, value=1024, step=1024)
            loss_masking = st.selectbox(label="Loss Calculation Style", options=["pretraining", "finetuning"])

        st.form_submit_button(label="Update Config")

    with st.form(key="tokenizer_form"):
        col1, col2 = st.columns(2)
        with col1:
            min_time_unit = st.number_input(
                label="Min Time Unit",
                min_value=0.01,
                value=0.01,
                step=0.01,
                format="%.2f",
            )
        with col2:
            n_velocity_bins = st.number_input(
                label="Velocity Bins",
                min_value=1,
                value=32,
                step=1,
            )

        st.form_submit_button(label="Update Tokenizer")

    config = {
        "base_dataset_name": base_dataset_name,
    }

    tokenizer_parameters = {
        "min_time_unit": min_time_unit,
        "n_velocity_bins": n_velocity_bins,
        "special_tokens": special_tokens,
    }

    tokenizer = ExponentialTokenizer(**tokenizer_parameters)
    dataset_name = "AugmentedDataset"
    dataset = load_hf_dataset(
        config=config,
        dataset_name=dataset_name,
        dataset_split=dataset_split,
    )
    midi_dataset = MedianDataset(
        dataset=dataset,
        tokenizer=tokenizer,
        sequence_length=sequence_length,
        notes_per_record=notes_per_record,
        loss_masking=loss_masking,
    )

    st.write(f"Total Samples: {len(midi_dataset)}")

    with st.expander("Configuration"):
        st.json(config)

    with st.expander("Tokenizer Parameters"):
        st.json(tokenizer_parameters)

    idx = st.number_input(
        label="Sample ID",
        value=0,
        min_value=0,
        max_value=len(midi_dataset) - 1,
    )
    sample = midi_dataset[idx]
    record_id, start = midi_dataset._index_to_record_and_start(idx=idx)
    st.write(record_id, start)
    st.json(dataset[record_id], expanded=False)

    with st.expander(label="Source Data"):
        st.json(sample["source"])

    extracted = sample["prediction_task"]
    st.write(f"Extraction Type: {extracted}")

    src_token_ids = sample["source_token_ids"]
    tgt_token_ids = sample["target_token_ids"]

    st.write("### Token Information")
    col1, col2 = st.columns(2)
    with col1:
        st.write("Source Tokens:")
        st.write(src_token_ids)
    with col2:
        st.write("Target Tokens:")
        st.write(tgt_token_ids)

    st.write("### Target Mask Visualization")
    target_mask_fig = plot_target_mask(sample["target_mask"])
    st.pyplot(target_mask_fig)

    # Display some statistics about the target mask
    true_count = torch.sum(sample["target_mask"]).item()
    total_count = len(sample["target_mask"])
    true_percentage = (true_count / total_count) * 100

    st.write("Target Mask Statistics:")
    st.write(f"- Total tokens: {total_count}")
    st.write(f"- Tokens used for loss calculation: {true_count}")
    st.write(f"- Percentage of tokens used: {true_percentage:.2f}%")

    src_tokens = [midi_dataset.tokenizer.vocab[token_id] for token_id in src_token_ids]

    low_median_token = "<LOW_FROM_MEDIAN>"
    high_median_token = "<HIGH_FROM_MEDIAN>"
    low_median_position = src_tokens.index(low_median_token)
    high_median_position = src_tokens.index(high_median_token)

    low_median_tokens = src_tokens[low_median_position:high_median_position]
    high_median_tokens = src_tokens[high_median_position:]

    low_median_notes = midi_dataset.tokenizer.untokenize(low_median_tokens)
    high_median_notes = midi_dataset.tokenizer.untokenize(high_median_tokens)

    low_median_piece = ff.MidiPiece(low_median_notes)
    high_median_piece = ff.MidiPiece(high_median_notes)

    st.write("### Visualizations")
    st.write("#### Combined View:")
    streamlit_pianoroll.from_fortepyan(piece=low_median_piece, secondary_piece=high_median_piece)

    col1, col2 = st.columns(2)
    with col1:
        st.write("#### Low Median:")
        streamlit_pianoroll.from_fortepyan(piece=low_median_piece)
    with col2:
        st.write("#### High Median:")
        streamlit_pianoroll.from_fortepyan(piece=high_median_piece)


if __name__ == "__main__":
    main()
