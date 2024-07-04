CREATE TABLE prompt_notes (
    prompt_id SERIAL PRIMARY KEY,
    start_time FLOAT,
    end_time FLOAT,
    composer VARCHAR(255) NULL,
    title VARCHAR(255) NULL,
    midi_filename VARCHAR(255) NULL,
    prompt_notes JSON,
    UNIQUE (start_time, end_time, midi_filename)
);
