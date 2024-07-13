import json
import hashlib

import torch
import pandas as pd

from gpt2.model import GPT
from artifacts import get_voice_range
from data.tokenizer import AwesomeTokenizer, ExponentialTokenizer


def prepare_prompts(
    record: dict,
    extraction_type: str,
    time_step: float,
    prompt_duration: float,
    target_context_duration: float,
) -> list[dict]:
    low, high = get_voice_range(voice=extraction_type)
    time = 0

    notes = pd.DataFrame(record["notes"])
    source = json.loads(record["source"])
    if "midi_filename" in source.keys():
        midi_name = source["midi_filename"]
    elif "youtube_id" in source.keys():
        midi_name = source["youtube_id"]
    else:
        midi_name = hashlib.sha256(record["source"])

    prompts = []
    while time + prompt_duration < notes.end.max():
        start = time
        end = time + prompt_duration

        fragment = notes[(notes.start > start) & (notes.end < end)]
        fragment_start = fragment.start.min()
        fragment_end = fragment.start.max()

        fragment.end -= fragment_start
        fragment.start -= fragment_start

        extracted_ids = (fragment.pitch >= low) & (fragment.pitch < high)
        source_notes = fragment[~extracted_ids]
        target_notes = fragment[extracted_ids]
        target_prompt = target_notes[target_notes.end < target_context_duration]
        if len(fragment) == 0:
            continue
        prompt = {
            "source_notes": source_notes,
            "target_prompt": target_prompt,
            "prompt_notes": pd.concat([source_notes, target_prompt]),
            "start_time": fragment_start,
            "end_time": fragment_end,
            "midi_name": midi_name,
            "source": record["source"],
        }
        prompts.append(prompt)
        time += time_step
    return prompts


def generate_bass(
    model: GPT,
    tokenizer: ExponentialTokenizer | AwesomeTokenizer,
    prompt_notes: pd.DataFrame,
    prompt_bass: pd.DataFrame,
    prompt_context_duration: float,
    target_context_duration: float,
    device: torch.device,
    temperature: float = 1.0,
    max_new_tokens: int = 512,
) -> pd.DataFrame:
    """
    Generate bass notes using the given model and tokenizer.

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
    prompt_notes = prompt_notes[prompt_notes.end < prompt_context_duration]
    prompt_bass = prompt_bass[prompt_bass.end < target_context_duration]

    # Handle the case where there's no target context
    if target_context_duration == 0:
        prompt_bass = pd.DataFrame(columns=prompt_notes.columns)

    # Tokenize prompt and target notes
    step_sequence = tokenizer.tokenize(prompt_notes)
    step_bass = tokenizer.tokenize(prompt_bass)
    # Combine prompt, bass marker, and target into input sequence
    input_sequence = ["<NO_BASS>"] + step_sequence + ["<BASS>"] + step_bass
    # Convert tokens to ids and prepare input tensor
    input_token_ids = torch.tensor(
        [[tokenizer.token_to_id[token] for token in input_sequence]],
        device=device,
    )

    # Generate new tokens using the model
    output = model.generate(
        idx=input_token_ids,
        temperature=temperature,
        max_new_tokens=max_new_tokens,
    )

    # Convert output to numpy array and decode tokens
    output = output[0].cpu().numpy()
    out_tokens = [tokenizer.vocab[token_id] for token_id in output]

    # Extract bass tokens (everything after the <BASS> marker)
    bass_command_position = out_tokens.index("<BASS>")
    bass_tokens = out_tokens[bass_command_position:].copy()

    # Convert bass tokens back to notes
    output_bass_notes = tokenizer.untokenize(bass_tokens)

    # Select only the newly generated notes within the prompt duration
    notes_after_context = output_bass_notes.start > target_context_duration
    notes_within_step = output_bass_notes.end < prompt_context_duration
    valid_new_notes = notes_after_context & notes_within_step
    bass_notes = output_bass_notes[valid_new_notes].copy()

    return bass_notes


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
        model (GPT): The GPT model for generation.
        tokenizer (ExponentialTokenizer | AwesomeTokenizer): The tokenizer for encoding/decoding notes.
        prompt_notes (pd.DataFrame): DataFrame containing prompt notes.
        target_notes (pd.DataFrame): DataFrame containing target notes.
        prompt_context_duration (float): Duration of the prompt context.
        target_context_duration (float): Duration of the target context.
        time_step (float): Time step for each iteration of generation.
        device (torch.device): The device to run the model on.
        temperature (float, optional): Temperature for sampling. Defaults to 1.0.
        max_new_tokens (int, optional): Maximum number of new tokens to generate. Defaults to 512.

    Returns:
        Tuple[pd.DataFrame, List[Tuple[ff.MidiPiece, ff.MidiPiece]]]:
            - DataFrame containing all generated bass notes.
            - List of tuples, each containing a pair of MidiPieces (source_piece, bass_prompt_piece) for debugging.

    The function generates bass notes iteratively, using the provided model and tokenizer.
    It processes the input in steps, generating new bass notes for each time step based on
    the given prompt and previously generated bass notes. The generation continues until
    the end of the prompt notes is reached.
    """
    # Initialize the first step with notes within the prompt and target context durations
    step_prompt_notes = prompt_notes[prompt_notes.end < prompt_context_duration].copy()
    step_bass_notes = target_notes[target_notes.end < target_context_duration].copy()
    # Initialize the list of all bass notes with the initial target notes
    all_bass_notes = [step_bass_notes]
    time = 0
    end = prompt_notes.end.max()

    # Handle the case where there's no target context
    if target_context_duration == 0:
        step_bass_notes = pd.DataFrame(columns=prompt_notes.columns)
    it = 0
    # Iterate through the piece, generating bass notes in steps
    while time + time_step <= end:
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
        input_sequence = ["<NO_BASS>"] + step_sequence + ["<BASS>"] + step_bass
        # Convert tokens to ids and prepare input tensor
        input_token_ids = torch.tensor(
            [[tokenizer.token_to_id[token] for token in input_sequence]],
            device=device,
        )
        # Generate new tokens using the model
        output = model.generate(
            idx=input_token_ids,
            temperature=temperature,
            max_new_tokens=max_new_tokens,
        )
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
        step_bass_notes = step_bass_notes[step_bass_notes.start > target_context_duration + time_step].copy()
        step_bass_notes.start -= target_context_duration + time_step
        step_bass_notes.end -= target_context_duration + time_step

    # Combine all generated bass notes and return
    return pd.concat(all_bass_notes).reset_index(drop=True)


def generate_from_prompt(
    model: GPT,
    tokenizer: AwesomeTokenizer | ExponentialTokenizer,
    prompt: dict,
    parameters: dict,
    device: torch.device,
):
    prompt_notes = pd.DataFrame(json.loads(prompt["prompt_notes"]))
    if parameters["task"] == "bass_prediction":
        low, high = get_voice_range("bass")

        no_bass_notes = prompt_notes[prompt_notes.pitch > high]
        target_notes = prompt_notes[prompt_notes.pitch < high]
        generated_notes = generate_bass_iteratively(
            model=model,
            tokenizer=tokenizer,
            prompt_notes=no_bass_notes,
            target_notes=target_notes,
            prompt_context_duration=parameters["prompt_context_duration"],
            target_context_duration=parameters["target_context_duration"],
            time_step=parameters["time_step"],
            device=device,
            temperature=parameters["temperature"],
            max_new_tokens=parameters["max_new_tokens"],
        )
    return generated_notes
