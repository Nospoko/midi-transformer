CREATE TABLE generation_parameters (
    id SERIAL PRIMARY KEY,
    temperature FLOAT,
    max_new_tokens INT,
    prompt_context_duration FLOAT,  -- setpoint value
    target_context_duration FLOAT DEFAULT 0,  -- does not really make sense in next token prediction task
    time_step FLOAT,
    task VARCHAR(255),  -- bass_prediction or next_token_prediction
    UNIQUE (temperature, max_new_tokens, prompt_context_duration, target_context_duration, task)
);
