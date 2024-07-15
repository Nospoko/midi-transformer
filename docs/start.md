## Startup
#### Install requirements
```shell
pip install -r requirements.txt
```
#### Environment
.env file should look like this:
```plaintext
WANDB_API_KEY=$YOUR_API_KEY
HF_TOKEN=$YOUR_AUTH_TOKEN
POSTGRES_PASSWORD=my_password
POSTGRES_USER=root
POSTGRES_HOST=localhost
POSTGRES_PORT=5432
HF_WRITE_TOKEN=optional_write_token
```
#### Run training
For example:
```shell
PYTHONPATH=. torchrun --nproc-per-node=2 \
gpt2/train.py --config-name=gpt2_pretraining \
data.batch_size=32 \
data.gradient_accumulation_steps=8 \
data.max_iters=12000 \
data.sequence_length=1024 \
dataset.notes_per_record=128 \
dataset.step=48 \
dataset.extra_datasets="['roszcz/giant-midi-sustain-v2']" \
dataset.augmentation.max_pitch_shift=0 \
dataset.augmentation.speed_change_factors=[] \
lr.warmup_iters=4000 \
lr.learning_rate=1e-4 \
lr.min_lr=1e-5 \
model=gpt2 \
system.dataloader_workers=44 \
system.compile=false \
loss_masking=pretrianing \
init_from=scratch
```
or
```shell
python -m gpt2.train --config-name=debugging system.device=cpu
```

#### Database
Use a database for storing model generations!
Start your local instance by running
```shell
docker-compose up
```
Example environment should enable you to use this database.

#### Run main dashboard
```shell
PYTHONPATH=. streamlit run --server.port 4567 dashboards/main.py
```

#### Validation examples
Define prompt + generation_parameters pairs in validation_examples table. They will be used
for generating sequences during training.

Navigate to "validation_examples_manager" display mode.

Choose "Generate prompts" on the sidebar and choose generation parameters.
Add prompts to validation table by clicking buttons underneath.

Browse the validation examples by choosing "Validation prompts" on the sidebar.

#### Generation during training
To generate examples during training, run
```shell
touch .generate
```
This will make the training script fetch validation examples and based on then,
push new generations to database.

#### Browsing generated examples
To browse examples navigate to "browse_generated" display mode on the dashboard
