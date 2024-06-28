import pandas as pd
import fortepyan as ff
import streamlit as st
import streamlit_pianoroll
from datasets import load_dataset


def main():
    dataset_names = [
        "BassExtractedDataset",
    ]
    dataset_name = st.selectbox(label="dataset", options=dataset_names)
    dataset_split = st.selectbox(label="split", options=["train", "test", "validation"])
    with st.form(key="config_form"):
        base_dataset_name = st.text_input(label="base_dataset_name", value="roszcz/maestro-sustain-v2")
        extra_datasets = st.text_input(label="extra_datasets (comma separated)", value="")
        notes_per_record = st.number_input(label="notes_per_record", min_value=1, value=60)
        step = st.number_input(label="step", min_value=1, value=60)
        pause_detection_threshold = st.number_input(label="pause_detection_threshold", value=4)

        st.form_submit_button(label="Submit")

    extra_datasets_list = [x.strip() for x in extra_datasets.split(",") if x.strip()]

    config = {
        "base_dataset_name": base_dataset_name,
        "extra_datasets": extra_datasets_list,
        "notes_per_record": notes_per_record,
        "step": step,
        "pause_detection_threshold": pause_detection_threshold,
    }

    dataset = load_dataset(
        f"midi_datasets/{dataset_name}",
        split=dataset_split,
        trust_remote_code=True,
        num_proc=8,
        **config,
    )

    total_tokens = config["notes_per_record"] * dataset.num_rows
    st.write(f"rows: {dataset.num_rows}")
    st.write(f"total notes: {total_tokens}")
    with st.expander("config"):
        st.write(config)

    idx = st.number_input(label="record_id", value=0, max_value=len(dataset))
    record = dataset[idx]

    with st.expander(label="source"):
        st.json(record["source"])

    source_notes = pd.DataFrame(record["source_notes"])
    target_notes = pd.DataFrame(record["target_notes"])
    src_piece = ff.MidiPiece(source_notes, source=record["source"])
    tgt_piece = ff.MidiPiece(target_notes, source=record["source"])
    st.write("#### Together:")
    streamlit_pianoroll.from_fortepyan(piece=src_piece, secondary_piece=tgt_piece)
    st.write("#### Prompt:")
    streamlit_pianoroll.from_fortepyan(piece=src_piece)
    st.write("#### Target:")
    streamlit_pianoroll.from_fortepyan(piece=tgt_piece)


if __name__ == "__main__":
    main()
