import streamlit as st
from datasets import load_dataset
import streamlit_pianoroll
import fortepyan as ff
from object_generators.tokenizer_generator import TokenizerGenerator

import json


def main():
    dataset_names = ["OneTimeTokDataset"]
    dataset_name = st.selectbox(label="dataset", options=dataset_names)
    dataset_split = st.selectbox(label="split", options=["train", "test", "validation"])
    
    if dataset_name == "OneTimeTokDataset":
        dataset = load_dataset("./OneTimeTokDataset", name="debugging", split=dataset_split)
    
    idx = st.number_input(label="record_id", value=0, max_value=len(dataset))
    record = dataset[idx]
    
    with st.expander(label="source"):
        st.json(record["source"])
        
    tokenzier_generator = TokenizerGenerator()
    tokenier_info = json.loads(dataset.description)
    tokenizer = tokenzier_generator.generate_tokenizer(
        name=tokenier_info["tokenizer_name"], 
        parameters=tokenier_info["tokenizer_parameters"],
    )
    
    notes = tokenizer.untokenize(record["note_tokens"])
    piece = ff.MidiPiece(notes, source=record["source"])

    streamlit_pianoroll.from_fortepyan(piece=piece)
    
if __name__ == "__main__":
    main()
