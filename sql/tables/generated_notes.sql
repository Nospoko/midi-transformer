CREATE TABLE generated_notes (
    generation_id SERIAL PRIMARY KEY,
    parameters_id INT REFERENCES generation_parameters(parameters_id),
    prompt_id INT REFERENCES prompt_notes(prompt_id),
    model_id INT REFERENCES models(model_id),
    generated_notes JSON  -- generated notes
);
