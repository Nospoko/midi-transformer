CREATE TABLE validation_prompts (
    id SERIAL PRIMARY KEY,
    parameters_id INT REFERENCES generation_parameters(id),
    prompt_id INT REFERENCES prompt_notes(id)
);
