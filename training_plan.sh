# Run this first and wait for the datasets to cache
python3.10 -m scripts.cache_datasets

# train on exponential datasets
python3.10 train.py --config-name=gpt2_small
python3.10 train.py --config-name=gpt2_pretraining model="gpt2"
python3.10 train.py --config-name=gpt2_pretraining model="gpt2_medium"

# train on awesome tokens datasets
python3.10 train.py --config-name=gpt2_pretraining model="gpt2_medium" data="awesome-giant"
python3.10 train.py --config-name=gpt2_pretraining model="gpt2" data="awesome-giant"
python3.10 train.py --config-name=gpt2_small data = "awesome-giant"

# train without augmentation
python3.10 train.py --config-name=gpt2_pretraining model="gpt2" data="awesome-giant" data.dataset_name: "giant-mid-coarse"
python3.10 train.py --config-name=gpt2_pretraining model="gpt2" data="exponential-giant" data.dataset_name: "giant-mid-coarse"


# Try training a large model
python3.10 train.py --config-name=gpt2_pretraining model="gpt2_large" data="awesome-colossal"
python3.10 train.py --config-name=gpt2_pretraining model="gpt2_large" data="exponential-colossal"

# Train a VERY large model
python3.10 train.py --config-name=gpt2_pretraining model="gpt2_xl" data="awesome-colossal"
