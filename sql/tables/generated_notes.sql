CREATE TABLE generated_notes (
    id SERIAL PRIMARY KEY,
    parameters_id INT REFERENCES generation_parameters(id),
    prompt_id INT REFERENCES prompt_notes(id),
    model_id INT REFERENCES models(id),
    generated_notes JSON,  -- generated notes
    UNIQUE(parameters_id, prompt_id, model_id)  -- One generations per parameters
);
