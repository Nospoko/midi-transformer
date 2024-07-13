import json

import pandas as pd
import sqlalchemy as sa

from data.database_connection import database_cnx

prompt_dtype = {
    "prompt_id": sa.Integer,
    "midi_name": sa.String(255),
    "start_time": sa.Float,
    "end_time": sa.Float,
    "source": sa.JSON,
    "prompt_notes": sa.JSON,
}

model_dtype = {
    "model_id": sa.Integer,
    "base_model_id": sa.Integer,
    "name": sa.String(255),
    "milion_parameters": sa.Integer,
    "best_val_loss": sa.Float,
    "total_tokens": sa.Integer,
    "configs": sa.JSON,
    "training_task": sa.String(255),
    "wandb_link": sa.Text,
}

parameter_dtype = {
    "parameters_id": sa.Integer,
    "temperature": sa.Float,
    "max_new_tokens": sa.Integer,
    "prompt_context_duration": sa.Float,
    "target_context_duration": sa.Float,
    "time_step": sa.Float,
    "task": sa.String(255),
}

generated_notes_dtype = {
    "generation_id": sa.Integer,
    "parameters_id": sa.Integer,
    "prompt_id": sa.Integer,
    "model_id": sa.Integer,
    "generated_notes": sa.JSON,
}

validation_examples_dtype = {
    "example_id": sa.Integer,
    "parameters_id": sa.Integer,
    "prompt_id": sa.Integer,
}


models_table = "models"
parameters_table = "generation_parameters"
generations_table = "generated_notes"
prompt_table = "prompt_notes"
validation_table = "validation_examples"


def insert_validation_generations_batch(generations: list[dict]):
    generation_data = []
    for generation_info in generations:
        param_id = generation_info["parameters_id"]
        prompt_id = generation_info["prompt_id"]
        model_id = generation_info["model_id"]
        query = f"""
        SELECT
            generation_id
        FROM
            {generations_table}
        WHERE
            parameters_id = {param_id}
        AND
            prompt_id = {prompt_id}
        AND
            model_id = {model_id}
        """
        existing_record = database_cnx.read_sql(sql=query)

        if existing_record.empty:
            generation_data.append(
                {
                    "parameters_id": param_id,
                    "prompt_id": prompt_id,
                    "model_id": model_id,
                    "generated_notes": generation_info["generated_notes"],
                }
            )

    if generation_data:
        df = pd.DataFrame(generation_data)

        # Insert the generated notes
        database_cnx.to_sql(
            df=df,
            table=generations_table,
            dtype=generated_notes_dtype,
            index=False,
            if_exists="append",
        )


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
    parameters_ids = [register_generation_parameters(param) for param in parameters]
    prompt_ids = [register_prompt_notes(prompt) for prompt in prompts]
    model_ids = [register_model(model) for model in models]

    generation_data = []

    for param_id, prompt_id, model_id, gen_notes in zip(parameters_ids, prompt_ids, model_ids, generated_notes_json):
        # Check if the record already exists
        query = f"""
        SELECT generation_id
        FROM generated_notes
        WHERE parameters_id = {param_id}
          AND prompt_id = {prompt_id}
          AND model_id = {model_id}
        """
        existing_record = database_cnx.read_sql(sql=query)

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
            table=generations_table,
            dtype=generated_notes_dtype,
            index=False,
            if_exists="append",
        )


def insert_validation_prompt(
    prompt: dict,
    parameters: dict,
):
    # Get or create IDs
    parameters_id = register_generation_parameters(parameters)
    prompt_id = register_prompt_notes(prompt)

    # Check if the record already exists
    query = f"""
    SELECT example_id
    FROM {validation_table}
    WHERE parameters_id = {parameters_id}
      AND prompt_id = {prompt_id}
    """

    existing_record = database_cnx.read_sql(sql=query)

    if existing_record.empty:
        validation_prompt_data = {
            "parameters_id": parameters_id,
            "prompt_id": prompt_id,
        }
        # Insert the generation data
        df = pd.DataFrame([validation_prompt_data])
        database_cnx.to_sql(
            df=df,
            table=validation_table,
            dtype=validation_examples_dtype,
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
    parameters_id = register_generation_parameters(parameters)
    prompt_id = register_prompt_notes(prompt)
    model_id = register_model(model_registration=model)

    # Check if the record already exists
    query = f"""
    SELECT generation_id
    FROM generated_notes
    WHERE parameters_id = {parameters_id}
      AND prompt_id = {prompt_id}
      AND model_id = {model_id}
    """

    existing_record = database_cnx.read_sql(sql=query)

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
            table=generations_table,
            dtype=generated_notes_dtype,
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
        {parameters_table}
    WHERE
        parameters_id = {parameters_id}
    """
    df = database_cnx.read_sql(sql=query)
    return df


def get_prompt(prompt_id: int) -> pd.DataFrame:
    query = f"""
    SELECT
        *
    FROM
        {prompt_table}
    WHERE
        prompt_id = {prompt_id}
    """
    df = database_cnx.read_sql(sql=query)
    return df


def get_prompts_for_model(model_id: int) -> pd.DataFrame:
    query = f"""
    SELECT DISTINCT
        pn.prompt_id,
        pn.midi_name,
        pn.start_time,
        pn.end_time,
        pn.dataset
    FROM {prompt_table} pn
    JOIN generated_notes gn ON pn.prompt_id = gn.prompt_id
    WHERE gn.model_id = {model_id}
    """
    df = database_cnx.read_sql(sql=query)

    # Fetch JSON fields separately
    if not df.empty:
        json_query = f"""
        SELECT prompt_id, source, {prompt_table}
        FROM {prompt_table}
        WHERE prompt_id IN ({','.join(map(str, df['prompt_id']))})
        """
        json_df = database_cnx.read_sql(sql=json_query)

        # Merge the results
        df = df.merge(json_df, on="prompt_id", how="left")

    return df


def get_parameters_for_model_and_prompt(model_id: int, prompt_id: int) -> pd.DataFrame:
    query = f"""
    SELECT DISTINCT gp.*
    FROM {parameters_table} gp
    JOIN generated_notes gn ON gp.parameters_id = gn.parameters_id
    WHERE gn.model_id = {model_id} AND gn.prompt_id = {prompt_id}
    """
    df = database_cnx.read_sql(sql=query)
    return df


def get_models(model_name: str) -> pd.DataFrame:
    query = f"""
    SELECT
        *
    FROM
        {models_table}
    WHERE
        name = '{model_name}'
    """
    df = database_cnx.read_sql(sql=query)
    if len(df) == 0:
        return {}
    return df


def get_model_id(model_name: str) -> int:
    query = f"""
    SELECT
        model_id
    FROM
        {models_table}
    WHERE
        name = '{model_name}'
    """
    df = database_cnx.read_sql(sql=query)
    if len(df) == 0:
        return None
    else:
        # Here, we are interested in the last recorded checkpoint
        return df.iloc[-1]["model_id"]


def purge_model(model_name: str):
    notes_query = f"""
    DELETE FROM {generations_table}
    WHERE model_id IN (
        SELECT model_id FROM {models_table} WHERE name = '{model_name}'
    )
    """
    database_cnx.execute(notes_query)

    model_query = f"""
    DELETE FROM {models_table}
    WHERE name = '{model_name}'
    """
    database_cnx.execute(model_query)


def remove_validation_prompt(validation_prompt_id: int):
    query = f"""
    DELETE FROM
        {validation_table}
    WHERE
        example_id = {validation_prompt_id}
    """
    database_cnx.execute(query=query)


def get_model_predictions(
    model_filters: dict = None,
    prompt_filters: dict = None,
    parameter_filters: dict = None,
) -> pd.DataFrame:
    base_query = f"""
    SELECT
        gn.*
    FROM
        generated_notes gn
    JOIN
        {models_table} m ON gn.model_id = m.model_id
    JOIN
        {prompt_table} pn ON gn.prompt_id = pn.prompt_id
    JOIN
        {parameters_table} gp ON gn.parameters_id = gp.parameters_id
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
    df = database_cnx.read_sql(sql=query)
    return df[column].dropna().tolist()


def get_all_models() -> pd.DataFrame:
    query = f"SELECT * FROM {models_table}"
    df = database_cnx.read_sql(sql=query)
    return df


def get_all_generation_parameters() -> pd.DataFrame:
    query = f"SELECT * FROM {parameters_table}"
    df = database_cnx.read_sql(sql=query)
    return df


def get_all_prompt_notes() -> pd.DataFrame:
    query = f"SELECT * FROM {prompt_table}"
    df = database_cnx.read_sql(sql=query)
    return df


def get_all_validation_prompts() -> pd.DataFrame:
    query = f"""
    SELECT
        *
    FROM
        {validation_table} vp
    JOIN
        {prompt_table} pn ON vp.prompt_id = pn.prompt_id
    JOIN
        {parameters_table} gp ON vp.parameters_id = gp.parameters_id
    """
    df = database_cnx.read_sql(sql=query)
    return df


def register_model(model_registration: dict) -> int:
    # Check if the record already exists
    query = f"""
    SELECT
        model_id
    FROM
        {models_table}
    WHERE
        name = '{model_registration['name']}'
    AND
        iter_num = {model_registration['iter_num']}
    AND
        training_task = '{model_registration['training_task']}'
    """

    existing_records = database_cnx.read_sql(sql=query)

    if not existing_records.empty:
        return existing_records.iloc[0]["model_id"]

    # If the record doesn't exist, insert it
    df = pd.DataFrame([model_registration])
    table = models_table
    database_cnx.to_sql(
        df=df,
        table=table,
        dtype=model_dtype,
        index=False,
        if_exists="append",
    )

    df = database_cnx.read_sql(sql=query)
    return df.iloc[0]["model_id"]


def register_model_from_checkpoint(
    checkpoint: dict,
    run_name: str,
):
    # Hard-coded for the specific naming style
    milion_parameters = run_name.split("-")[2][:-1]
    init_from = checkpoint["config"]["init_from"]
    base_model_id = None
    if init_from != "scratch":
        base_model_id = get_model_id(model_name=init_from)

    model_registration = {
        "name": run_name,
        "milion_parameters": milion_parameters,
        "best_val_loss": float(checkpoint["best_val_loss"]),
        "iter_num": checkpoint["iter_num"],
        "training_task": checkpoint["config"]["task"],
        "configs": checkpoint["config"],
    }
    if "wandb" in checkpoint.keys():
        model_registration |= {"wandb_link": checkpoint["wandb"]}
    if "total_tokens" in checkpoint.keys():
        model_registration |= {"total_tokens": checkpoint["total_tokens"]}
    if base_model_id is not None:
        model_registration |= {"base_model_id": base_model_id}

    model_id = register_model(model_registration=model_registration)

    return model_registration, model_id


def register_generation_parameters(generation_parameters: dict):
    # Check if the record already exists
    query = f"""
    SELECT parameters_id
    FROM {parameters_table}
    WHERE temperature = {generation_parameters['temperature']}
      AND max_new_tokens = {generation_parameters['max_new_tokens']}
      AND prompt_context_duration = {generation_parameters['prompt_context_duration']}
      AND target_context_duration = {generation_parameters['target_context_duration']}
      AND task = '{generation_parameters['task']}'
    """
    existing_records = database_cnx.read_sql(sql=query)

    if not existing_records.empty:
        return existing_records.iloc[0]["parameters_id"]

    # Create DataFrame from parameters
    df = pd.DataFrame([generation_parameters])
    # Insert new record
    table = parameters_table
    database_cnx.to_sql(
        df=df,
        table=table,
        dtype=parameter_dtype,
        index=False,
        if_exists="append",
    )
    df = database_cnx.read_sql(sql=query)
    return df.iloc[0]["parameters_id"]


def register_prompt_notes(prompt_notes: dict):
    prompt_notes["prompt_notes"] = json.dumps(prompt_notes["prompt_notes"])

    # Check if the record already exists
    query = f"""
    SELECT prompt_id
    FROM {prompt_table}
    WHERE start_time = {prompt_notes['start_time']}
      AND end_time = {prompt_notes['end_time']}
      AND midi_name = '{prompt_notes['midi_name']}'
    """
    existing_records = database_cnx.read_sql(sql=query)

    if not existing_records.empty:
        return existing_records.iloc[0]["prompt_id"]

    # Create DataFrame from prompt_note
    df = pd.DataFrame([prompt_notes])
    # Insert new record
    table = prompt_table
    database_cnx.to_sql(
        df=df,
        table=table,
        dtype=prompt_dtype,
        index=False,
        if_exists="append",
    )
    df = database_cnx.read_sql(sql=query)
    return df.iloc[0]["prompt_id"]
