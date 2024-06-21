import fortepyan as ff
import streamlit as st
import streamlit_pianoroll
from datasets import load_dataset
from midi_tokenizers import ExponentialTimeTokenizer

from artifacts import special_tokens
from data.next_token_dataset import NextTokenDataset


def main():
    dataset_names = [
        "MidiSequenceDataset",
    ]
    dataset_name = st.selectbox(label="dataset", options=dataset_names)
    dataset_split = st.selectbox(label="split", options=["train", "test", "validation"])
    with st.form(key="config_form"):
        base_dataset_name = st.text_input(label="base_dataset_name", value="roszcz/maestro-sustain-v2")
        extra_datasets = st.text_input(label="extra_datasets (comma separated)", value="")
        notes_per_record = st.number_input(label="notes_per_record", min_value=1, value=60)
        step = st.number_input(label="step", min_value=1, value=60)
        pause_detection_threshold = st.number_input(label="pause_detection_threshold", value=4)
        sequence_length = st.number_input(label="sequence_length", min_value=1, value=5000, step=500)

        st.form_submit_button(label="Submit")

    with st.form(key="tokenizer_form"):
        min_time_unit = st.number_input(label="min_time_unit", min_value=0.01, value=0.01, step=0.01, format="%.2f")
        n_velocity_bins = st.number_input(label="n_velocity_bins", min_value=1, value=32, step=1)

        st.form_submit_button(label="Submit")

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

    tokenizer = ExponentialTimeTokenizer(**tokenizer_parameters)

    dataset = load_dataset(
        f"midi_datasets/{dataset_name}",
        split=dataset_split,
        trust_remote_code=True,
        num_proc=8,
        **config,
    )
    midi_dataset = NextTokenDataset(
        dataset=dataset,
        tokenizer=tokenizer,
        sequence_length=sequence_length,
    )
    total_tokens = config["notes_per_record"] * dataset.num_rows
    st.write(f"rows: {dataset.num_rows}")
    st.write(f"total notes: {total_tokens}")
    with st.expander("config"):
        st.write(config)

    idx = st.number_input(label="record_id", value=0, max_value=len(dataset))
    record = midi_dataset[idx]

    with st.expander(label="source"):
        st.json(record["source"])

    src_token_ids = record["source_token_ids"]
    tgt_token_ids = record["target_token_ids"]

    src_tokens = [midi_dataset.tokenizer.vocab[token_id] for token_id in src_token_ids]
    tgt_tokens = [midi_dataset.tokenizer.vocab[token_id] for token_id in tgt_token_ids]

    source_notes = midi_dataset.tokenizer.untokenize(src_tokens)
    target_notes = midi_dataset.tokenizer.untokenize(tgt_tokens)

    src_piece = ff.MidiPiece(source_notes)
    tgt_piece = ff.MidiPiece(target_notes)

    token_columns = st.columns(2)
    token_columns[0].write(src_tokens)
    token_columns[1].write(tgt_tokens)
    st.write("#### Prompt:")
    streamlit_pianoroll.from_fortepyan(piece=src_piece)
    st.write("#### Target:")
    streamlit_pianoroll.from_fortepyan(piece=tgt_piece)


if __name__ == "__main__":
    main()
