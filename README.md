# Emocontext 2019
Submission to EmoContext 2019

## Sentiment/Emotion Lexicons
https://drive.google.com/open?id=1cTDBqRjFAy4LchXrCHhMEjcFvzE4I1zn

## Instructions

### Install prerequisites
```bash
pip install -r requirements.txt
```

### Train model
```bash
./scripts/run.sh \
--mode TRAIN 
--log-level "DEBUG"
--train-file-path data/raw/train.txt \
--epochs ${NUM_EPOCHS}
```

This will produce a folder like `models/yyyymmddhhmmss`.
This folder path should be used as the ${MODEL_SAVE_DIRECTORY} variable for inference.


### Infer emotion
```bash
./scripts/run.sh \
--mode INFER \
--log-level "DEBUG" \
--test-file-path data/raw/devwithoutlabels.txt \
--model-directory ${MODEL_SAVE_DIRECTORY}
```
