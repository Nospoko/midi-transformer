CREATE TABLE models (
    model_id SERIAL PRIMARY KEY,
    name VARCHAR(255),
    num_parameters INT,
    best_val_loss FLOAT,
    total_tokens BIGINT,  -- total number of tokens the model trained on
    wandb_link TEXT
);
