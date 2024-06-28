# MIDI Transformer

## Table of Contents
1. [Overview](#overview)
2. [Features](#features)
3. [Installation](#installation)
4. [Running the Dashboards](#running-the-dashboards)
5. [Available Dashboards](#available-dashboards)
6. [Project Purpose](#project-purpose)
7. [How to Train Your Own Model](#how-to-train-your-own-model)
   - [Setup](#setup)
   - [Train the Model](#train-the-model)
   - [Use AwesomeTokensDataset](#use-awesometokensdataset)
   - [Augmentation](#augmentation)
8. [Dataset Sizes](#dataset-sizes)
8. [Important Links](#important-links)
9. [Code Style](#code-style)

## Overview
This project aims to model the emotional nuances of piano performances using GPT-2 transformer architectures. By leveraging MIDI format, the project trains models to generate and interpret musical performances, capturing the expressive qualities of piano music.

## Features
- **Streamlit Dashboards**: Visualize and interact with MIDI data and model outputs.
- **Model Training**: Train GPT-2 models to generate piano music sequences.
- **Data Augmentation**: Techniques such as pitch shifting and tempo changes to enhance training data.

## Installation
To get started, clone the repository and install the required dependencies:

```bash
git clone https://github.com/Nospoko/midi-transformer.git
cd midi-transformer
pip install -r requirements.txt
```

## Running the Dashboards
Dashboards are built with Streamlit. To run the dashboard:

```bash
PYTHONPATH=. streamlit run --server.port 4002 dashboards/main.py
```

## Available Dashboards
- **Main Dashboard**: `dashboards/main.py`
  - Visualize MIDI data and model predictions.
- **GPT Review Dashboard**: `dashboards/gpt_review.py`
  - Review and GPT model outputs.
- **MIDI Dataset Review Dashboard**: `dashboards/midi_dataset_review.py`
  - Explore and analyze the MIDI torch dataset.
- **HF Dataset Review Dashboard**: `dashboards/hf_dataset_review.py`
  - Review Hugging Face datasets defined in `tokenzied_midi_dataset` module.
- **Run Evaluation Dashboard**: `dashboards/run_eval.py`
  - Evaluate model performance.
- **Browse Generated Dashboard**: `dashboards/browse_generated.py`
  - Browse MIDI sequences generated with `python -m scripts.generate_all`.
- **Augmentation Review Dashboard**: `dashboards/augmentation_review.py`
  - Review data augmentation techniques.

## Project Purpose
This project explores the intersection of music and machine learning by:
- Modeling the expressive nuances of piano performances.
- Developing methods for data augmentation and MIDI data processing.
- Training transformer models to generate and interpret piano music.

## How to Train Your Own Model
### Setup
Ensure you have set up your environment correctly. Follow the `.env.example` pattern to set up your `.env` file with your personal tokens for wandb and Hugging Face.

### Train the Model
To train the model, use the following command:

```bash
python -m gpt2.train
```

### Use AwesomeTokensDataset
To use the `AwesomeTokensDataset`, first run:

```bash
python -m scripts.train_awesome_tokenizer
```

This creates a pre-trained tokenizer JSON in the `pretrained/awesome_tokenizers` directory.

### Augmentation
We use pitch_shift and change_speed augmentation techniques, sequentially (pitch_shift, then change_speed).

### Dataset Sizes

| Dataset                        | Train tokens | Test tokens | Validation tokens |
|--------------------------------|--------------|-------------|-------------------|
| Basic AwesomeTokensDataset     | 9,674,752   | 1,347,584  | 1,110,528        |
| Giant AwesomeTokensDataset     | 78,107,136   | 1,347,584  | 1,110,528          |
| Basic ExponentialTimeTokenDataset |    25,981,952 |     3,607,040  | 2,966,016   |
| Giant ExponentialTimeTokenDataset |   206,0451,84 |    3,607,040  | 2,966,016   |

---

## Important Links
- **Maestro Dataset**: [Link to dataset](https://magenta.tensorflow.org/datasets/maestro)
- **GitHub Repository**: [midi-transformer](https://github.com/Nospoko/midi-transformer)
- **Midi Tokenizers Repository**: [midi-tokenizers](https://github.com/Nospoko/midi-tokenizers)
- **Platform for pianists and algorithmic music enthusiasts**: [pianoroll.io](https://pianoroll.io)
### Code Style

This repository uses pre-commit hooks with forced python formatting ([black](https://github.com/psf/black),
[flake8](https://flake8.pycqa.org/en/latest/), and [isort](https://pycqa.github.io/isort/)):

```sh
pip install pre-commit
pre-commit install
```

Whenever you execute `git commit` the files altered / added within the commit will be checked and corrected.
`black` and `isort` can modify files locally - if that happens you have to `git add` them again.
You might also be prompted to introduce some fixes manually.

To run the hooks against all files without running `git commit`:

```sh
pre-commit run --all-files
```
