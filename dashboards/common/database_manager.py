import json

import pandas as pd
import sqlalchemy as sa

from dashboards.common.runtime import database_cnx  # Adjust the import based on your project structure

# Define dtypes
parameter_dtype = {
    "id": sa.Integer,
    "temperature": sa.Float,
    "max_new_tokens": sa.Integer,
    "prompt_context_duration": sa.Float,
    "target_context_duration": sa.Float,
    "task": sa.String(255),
}
prompt_dtype = {
    "id": sa.Integer,
    "start_time": sa.Float,
    "end_time": sa.Float,
    "composer": sa.String(255),
    "title": sa.String(255),
    "midi_filename": sa.String(255),
    "prompt_notes": sa.JSON,
}
model_dtype = {
    "id": sa.Integer,
    "name": sa.String(255),
    "num_parameters": sa.Integer,
    "best_val_loss": sa.Float,
    "total_tokens": sa.BigInteger,
    "wandb_link": sa.Text,
}
generated_notes_dtype = {
    "id": sa.Integer,
    "parameters_id": sa.Integer,
    "prompt_id": sa.Integer,
    "model_id": sa.Integer,
    "generated_notes": sa.JSON,
}


def get_or_create_id(table: str, filters: dict, dtype: dict) -> int:
    # Build the SELECT query to check if the record exists
    query = f"SELECT id FROM {table} WHERE 1=1"

    for key, value in filters.items():
        if "notes" in key:  # ignore the large fields
            continue
        if isinstance(value, str):
            query += f" AND {key} = '{value}'"
        else:
            query += f" AND {key} = {value}"

    df = database_cnx.read_sql(sql=query)

    if not df.empty:
        return df.iloc[0]["id"]

    # If the record does not exist, insert it and return the new ID
    df = pd.DataFrame([filters])
    database_cnx.to_sql(
        df=df,
        table=table,
        dtype=dtype,
        index=False,
        if_exists="append",
    )

    df = database_cnx.read_sql(sql=query)
    return df.iloc[0]["id"]


def insert_generated_notes_batch(
    models: list[dict],
    prompts: list[dict],
    parameters: list[dict],
    generated_notes: list[pd.DataFrame],
):
    generated_notes_json = [notes.to_json() for notes in generated_notes]

    # Convert prompt notes to JSON strings
    for prompt in prompts:
        prompt["prompt_notes"] = prompt["prompt_notes"].to_json()

    # Get or create IDs for all entries
    parameters_ids = [get_or_create_id("generation_parameters", param, parameter_dtype) for param in parameters]
    prompt_ids = [get_or_create_id("prompt_notes", prompt, prompt_dtype) for prompt in prompts]
    model_ids = [get_or_create_id("models", model, model_dtype) for model in models]

    generation_data = []

    for param_id, prompt_id, model_id, gen_notes in zip(parameters_ids, prompt_ids, model_ids, generated_notes_json):
        # Check if the record already exists
        query = f"""
        SELECT id
        FROM generated_notes
        WHERE parameters_id = {param_id}
          AND prompt_id = {prompt_id}
          AND model_id = {model_id}
        """
        existing_record = database_cnx.read_df(query)

        if existing_record.empty:
            generation_data.append(
                {
                    "parameters_id": param_id,
                    "prompt_id": prompt_id,
                    "model_id": model_id,
                    "generated_notes": gen_notes,
                }
            )

    # Create a DataFrame from the generation data
    if generation_data:
        df = pd.DataFrame(generation_data)

        # Insert the generated notes
        database_cnx.to_sql(
            df=df,
            table="generated_notes",
            dtype=generated_notes_dtype,
            index=False,
            if_exists="append",
        )


def insert_generated_notes(
    model: dict,
    prompt: dict,
    parameters: dict,
    generated_notes: pd.DataFrame,
):
    generated_notes = generated_notes.to_json()
    prompt["prompt_notes"] = prompt["prompt_notes"].to_json()

    # Get or create IDs
    parameters_id = get_or_create_id("generation_parameters", parameters, parameter_dtype)
    prompt_id = get_or_create_id("prompt_notes", prompt, prompt_dtype)
    model_id = get_or_create_id("models", model, model_dtype)

    # Check if the record already exists
    query = f"""
    SELECT id
    FROM generated_notes
    WHERE parameters_id = {parameters_id}
      AND prompt_id = {prompt_id}
      AND model_id = {model_id}
    """
    existing_record = database_cnx.read_df(query)

    if existing_record.empty:
        generation_data = {
            "parameters_id": parameters_id,
            "prompt_id": prompt_id,
            "model_id": model_id,
            "generated_notes": generated_notes,
        }
        # Insert the generation data
        df = pd.DataFrame([generation_data])
        database_cnx.to_sql(
            df=df,
            table="generated_notes",
            dtype={
                "id": sa.Integer,
                "parameters_id": sa.Integer,
                "prompt_id": sa.Integer,
                "model_id": sa.Integer,
                "generated_notes": sa.JSON,
            },
            index=False,
            if_exists="append",
        )


def register_model(model_registration: dict):
    df = pd.DataFrame([model_registration])

    table = "models"
    database_cnx.to_sql(
        df=df,
        table=table,
        dtype=model_dtype,
        index=False,
        if_exists="append",
    )


def insert_data(df: pd.DataFrame, table: str):
    database_cnx.to_sql(
        df=df,
        table=table,
        index=False,
        if_exists="append",
    )


def get_parameters(parameters_id: int) -> pd.DataFrame:
    query = f"""
    SELECT
        *
    FROM
        generation_parameters
    WHERE
        id = {parameters_id}
    """
    df = database_cnx.read_sql(sql=query)
    return df


def get_prompt(prompt_id: int) -> pd.DataFrame:
    query = f"""
    SELECT
        *
    FROM
        prompt_notes
    WHERE
        id = {prompt_id}
    """
    df = database_cnx.read_sql(sql=query)
    return df


def get_model_info(model_name: str) -> dict:
    query = f"""
        SELECT
            *
        FROM
            models
        WHERE
            name = '{model_name}'
    """
    df = database_cnx.read_df(query)
    if len(df) == 0:
        return {}
    res = df.iloc[0].to_dict()
    return res


def purge_model(model_name: str):
    query_table = f"""
    DELETE FROM generated_notes
    WHERE model_id IN (
        SELECT model_id FROM models WHERE name = '{model_name}'
    )
    """
    database_cnx.execute(query_table)

    model_query = f"""
    DELETE FROM models
    WHERE name = '{model_name}'
    """
    database_cnx.execute(model_query)


def get_model_predictions(
    model_filters: dict = None,
    prompt_filters: dict = None,
    parameter_filters: dict = None,
) -> pd.DataFrame:
    base_query = """
    SELECT
        gn.*
    FROM
        generated_notes gn
    JOIN
        models m ON gn.model_id = m.id
    JOIN
        prompt_notes pn ON gn.prompt_id = pn.id
    JOIN
        generation_parameters gp ON gn.parameters_id = gp.id
    WHERE
        1=1
    """

    # Apply model filters
    if model_filters:
        for key, value in model_filters.items():
            base_query += f" AND m.{key} = '{value}'"

    # Apply prompt filters
    if prompt_filters:
        for key, value in prompt_filters.items():
            base_query += f" AND pn.{key} = '{value}'"

    # Apply parameter filters
    if parameter_filters:
        for key, value in parameter_filters.items():
            base_query += f" AND gp.{key} = '{value}'"

    df = database_cnx.read_sql(sql=base_query)

    return df


def get_unique_values(column, table):
    query = f"SELECT DISTINCT {column} FROM {table} ORDER BY {column}"
    df = database_cnx.read_df(query)
    return df[column].dropna().tolist()


def get_common_prompts_and_parameters_for_models(model_id_1: int, model_id_2: int):
    query = f"""
    SELECT DISTINCT
        gn1.prompt_id,
        gn1.parameters_id
    FROM
        generated_notes gn1
    JOIN
        generated_notes gn2 ON gn1.prompt_id = gn2.prompt_id AND gn1.parameters_id = gn2.parameters_id
    WHERE
        gn1.model_id = {model_id_1} AND gn2.model_id = {model_id_2}
    """
    df = database_cnx.read_df(query)
    return df["prompt_id"].unique().tolist(), df["parameters_id"].unique().tolist()


def get_all_models() -> pd.DataFrame:
    query = "SELECT * FROM models"
    df = database_cnx.read_df(query)
    return df


def get_all_generation_parameters() -> pd.DataFrame:
    query = "SELECT * FROM generation_parameters"
    df = database_cnx.read_df(query)
    return df


def get_all_prompt_notes() -> pd.DataFrame:
    query = "SELECT * FROM prompt_notes"
    df = database_cnx.read_df(query)
    return df


def register_generation_parameters(generation_parameters: dict):
    # Create DataFrame from parameters
    df = pd.DataFrame([generation_parameters])

    # Check if the record already exists
    query = f"""
        SELECT id
        FROM generation_parameters
        WHERE temperature = {generation_parameters['temperature']}
          AND max_new_tokens = {generation_parameters['max_new_tokens']}
          AND prompt_context_duration = {generation_parameters['prompt_context_duration']}
          AND target_context_duration = {generation_parameters['target_context_duration']}
          AND task = '{generation_parameters['task']}'
    """
    existing_records = database_cnx.read_df(query)

    if not existing_records.empty:
        return existing_records.iloc[0]["id"]

    # Insert new record
    table = "generation_parameters"
    database_cnx.to_sql(
        df=df,
        table=table,
        dtype=parameter_dtype,
        index=False,
        if_exists="append",
    )
    return None


def register_prompt_notes(prompt_notes: dict):
    prompt_notes["prompt_notes"] = json.dumps(prompt_notes["prompt_notes"])

    # Create DataFrame from prompt_note
    df = pd.DataFrame([prompt_notes])

    # Check if the record already exists
    query = f"""
        SELECT id
        FROM prompt_notes
        WHERE start_time = {prompt_notes['start_time']}
          AND end_time = {prompt_notes['end_time']}
          AND composer = '{prompt_notes['composer']}'
          AND title = '{prompt_notes['title']}'
          AND midi_filename = '{prompt_notes['midi_filename']}'
    """
    existing_records = database_cnx.read_df(query)

    if not existing_records.empty:
        return existing_records.iloc[0]["id"]

    # Insert new record
    table = "prompt_notes"
    database_cnx.to_sql(
        df=df,
        table=table,
        dtype=prompt_dtype,
        index=False,
        if_exists="append",
    )
    return None
