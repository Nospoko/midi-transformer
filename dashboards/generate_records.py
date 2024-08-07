import os
from glob import glob
from contextlib import AbstractContextManager, nullcontext

import yaml
import torch
import streamlit as st
from datasets import Dataset

from gpt2.model import GPT
import gpt2.generation as generation
import data.database_manager as database_manager
import dashboards.common.utils as dashboard_utils
from dashboards.common.utils import select_generation_parameters
from data.tokenizer import AwesomeTokenizer, ExponentialTokenizer
from gpt2.utils import load_cfg, load_tokenizer, initialize_model


def multiselect_part_dataset(midi_dataset: Dataset) -> Dataset:
    source_df = midi_dataset.to_pandas()
    source_df["source"] = source_df["source"].map(yaml.safe_load)
    source_df["composer"] = [source["composer"] for source in source_df.source]
    source_df["title"] = [source["title"] for source in source_df.source]

    composers = source_df.composer.unique()
    selected_composers = st.multiselect(
        "Select composers",
        options=composers,
        default=composers[:3],
    )

    ids = source_df.composer.isin(selected_composers)
    piece_titles = source_df[ids].title.unique()
    selected_titles = st.multiselect(
        "Select titles",
        options=piece_titles,
        default=piece_titles[:3],
    )

    ids = source_df.composer.isin(selected_composers) & source_df.title.isin(selected_titles)
    return midi_dataset.select(source_df[ids].index.values)


def load_model_and_tokenizer():
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

        run_name = os.path.basename(checkpoint_path)
        model_registration, _ = database_manager.register_model_from_checkpoint(
            checkpoint=checkpoint,
            run_name=run_name,
        )

        st.success(f"Model loaded! Best validation loss: {checkpoint['best_val_loss']:.4f}")
        if "wandb" in checkpoint:
            st.link_button(label="View Training Run", url=checkpoint["wandb"])

    cfg = load_cfg(checkpoint=checkpoint)
    tokenizer = load_tokenizer(cfg=cfg)
    st.write("Training config")
    st.json(checkpoint["config"], expanded=False)
    ptdtype = {"float32": torch.float32, "bfloat16": torch.bfloat16, "float16": torch.float16}[cfg.system.dtype]
    device_type = "cuda" if "cuda" in device else "cpu"
    ctx = nullcontext() if device_type == "cpu" else torch.amp.autocast(device_type=device_type, dtype=ptdtype)

    return cfg, checkpoint, tokenizer, device, ctx, model_registration


def dataset_configuration():
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
        prediction_task = st.selectbox(
            "Extraction Type",
            options=["bass_prediction", "reverse_bass_prediction", "high_median_prediction", "-"],
            help="Select the type of notes to extract",
        )

    with st.spinner("Loading dataset..."):
        dataset = dashboard_utils.load_hf_dataset(dataset_path=dataset_path, dataset_split=dataset_split)
        dataset = multiselect_part_dataset(midi_dataset=dataset)

    st.success(f"Dataset loaded! Total records: {len(dataset)}")
    return dataset, prediction_task


def generate_music(
    dataset: Dataset,
    generation_parameters: dict,
    model: GPT,
    tokenizer: ExponentialTokenizer | AwesomeTokenizer,
    device: torch.device,
    ctx: AbstractContextManager,
    model_registration: dict,
    prompt_duration: float,
    prompt_creation_time_step: float,
    prediction_task: str = None,
):
    task = generation_parameters["task"]
    prompts = []

    with st.spinner("Slicing the records into prompts"):
        for record in dataset:
            if task == "next_token_prediction":
                prompts += generation.prepare_next_token_prediction_prompts(
                    record=record,
                    prompt_duration=prompt_duration,
                    time_step=prompt_creation_time_step,
                )
            elif task == "high_median_prediction":
                prompts += generation.prepare_high_median_prompts(
                    record=record,
                    prompt_duration=prompt_duration,
                    target_context_duration=generation_parameters["target_context_duration"],
                    time_step=prompt_creation_time_step,
                )
            else:
                prompts += generation.prepare_subsequence_prediction_prompts(
                    record=record,
                    prediction_task=prediction_task,
                    prompt_duration=prompt_duration,
                    time_step=prompt_creation_time_step,
                    target_context_duration=generation_parameters["target_context_duration"],
                )

    num_prompts = len(prompts)

    for idx, prompt in enumerate(prompts):
        with st.spinner(f"Generating... {idx + 1} / {num_prompts}"):
            if task == "next_token_prediction":
                source_notes = prompt["prompt_notes"]
                target_notes = generation.generate_continuation(
                    model=model,
                    tokenizer=tokenizer,
                    prompt_notes=source_notes,
                    prompt_context_duration=generation_parameters["prompt_context_duration"],
                    device=device,
                    ctx=ctx,
                    temperature=generation_parameters["temperature"],
                    max_new_tokens=generation_parameters["max_new_tokens"],
                )
                generated_notes = target_notes
            else:
                source_notes = prompt.pop("source_notes")
                target_prompt = prompt.pop("target_prompt")
                target_notes = generation.generate_subsequence_iteratively(
                    model=model,
                    tokenizer=tokenizer,
                    prompt_notes=source_notes,
                    target_notes=target_prompt,
                    prompt_context_duration=generation_parameters["prompt_context_duration"],
                    target_context_duration=generation_parameters["target_context_duration"],
                    time_step=generation_parameters["time_step"],
                    device=device,
                    max_new_tokens=generation_parameters["max_new_tokens"],
                    temperature=generation_parameters["temperature"],
                    ctx=ctx,
                )
                generated_notes = target_notes.iloc[len(target_prompt) :]

            database_manager.insert_generated_notes(
                model=model_registration,
                prompt=prompt,
                parameters=generation_parameters,
                generated_notes=generated_notes,
            )

    st.success(f"Generated {num_prompts} musical pieces")


def main():
    st.title("🎵 Music Generation Dashboard for populating the database")

    cfg, checkpoint, tokenizer, device, ctx, model_registration = load_model_and_tokenizer()

    tab1, tab2, tab3 = st.tabs(["Dataset Selection", "Generation Parameters", "Results"])

    with tab1:
        dataset, prediction_task = dataset_configuration()

    with tab2:
        run, generation_parameters, prompt_duration, prompt_creation_time_step = select_generation_parameters()

    if run:
        with tab3:
            st.header("Generation state")
            model = initialize_model(
                cfg,
                checkpoint=checkpoint,
                device=device,
                pad_token_id=tokenizer.pad_token_id,
            )
            generate_music(
                dataset=dataset,
                generation_parameters=generation_parameters,
                model=model,
                tokenizer=tokenizer,
                device=device,
                ctx=ctx,
                model_registration=model_registration,
                prompt_duration=prompt_duration,
                prompt_creation_time_step=prompt_creation_time_step,
                prediction_task=prediction_task,
            )


if __name__ == "__main__":
    main()
