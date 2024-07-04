CREATE TABLE generation_parameters (
    parameters_id SERIAL PRIMARY KEY,
    temperature FLOAT,
    max_new_tokens INT,
    prompt_context_duration FLOAT,  -- setpoint value. (notes.end - notes.start) could be < prompt_context_duration
    target_context_duration FLOAT DEFAULT 0,
    task VARCHAR(255),  -- bass_prediction or next_token_prediction
    UNIQUE (temperature, max_new_tokens, prompt_context_duration, target_context_duration, task)
);
