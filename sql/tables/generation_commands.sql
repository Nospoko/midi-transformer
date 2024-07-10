CREATE TABLE generation_commands (
    id SERIAL PRIMARY KEY,
    parameters_id INT REFERENCES generation_parameters(id),
    prompt_id INT REFERENCES prompt_notes(id),
    model_name VARCHAR(255),
    UNIQUE(parameters_id, prompt_id, model_name)  -- One generations per parameters
);
