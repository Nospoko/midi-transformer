CREATE TABLE generation_parameters (
    prompt_id SERIAL PRIMARY KEY,
    start_time FLOAT,
    end_time FLOAT,
    composer VARCHAR(255) NULL,
    title VARCHAR(255) NULL,
    midi_filename VARCHAR(255) NULL,
    prompt_context_duration FLOAT,  -- setpoint value. end - start could be < prompt_context_duration
    task VARCHAR(255),  -- bass_prediction or next_token_prediction
    prompt_notes JSON
);
