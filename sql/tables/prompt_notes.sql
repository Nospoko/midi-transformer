CREATE TABLE prompt_notes (
    prompt_id SERIAL PRIMARY KEY,
    midi_name VARCHAR(255),  -- youtube_id or midi_filename
    start_time FLOAT,
    end_time FLOAT,
    dataset VARCHAR(255) NULL,
    source JSON,
    prompt_notes JSON,
    UNIQUE (start_time, end_time, midi_name)
);
