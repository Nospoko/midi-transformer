CREATE TABLE models (
    id SERIAL PRIMARY KEY,
    name VARCHAR(255),
    milion_parameters INT,
    best_val_loss FLOAT,
    wandb_link TEXT
);
