CREATE TABLE generated_notes (
    generation_id SERIAL PRIMARY KEY,
    prompt_id INT REFERENCES generation_parameters(prompt_id),
    model_id INT REFERENCES models(model_id),
    generated_notes JSON  -- generated notes
);
