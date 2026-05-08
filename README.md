# Pixels to Predictions: DL Vision Challenge

## Overview
This repository contains the training and inference code for the Pixels to Predictions Kaggle competition (DL Spring 2026). The goal is to fine-tune SmolVLM-500M-Instruct to answer scientific multiple-choice questions given an image and textual context.

**Final Kaggle Public Score: 0.81287**

## Repository Structure

## Model Weights
Pre-trained LoRA adapter weights are available at:
https://drive.google.com/drive/folders/1uoTS1YctCeJKCaTOzDN2ZTFgRrTOELZo?usp=sharing

## Requirements
- Google Colab (GPU: T4 or above)
- Python 3.10+
- transformers
- peft
- accelerate
- torchao

Install dependencies by running Cell 1 of the notebook.

## How to Reproduce

1. Upload the competition dataset zip file to Google Drive at:
   `MyDrive/pixels-to-predictions.zip`

2. Open `pixels_to_predictions.ipynb` in Google Colab

3. Run all cells in order:
   - **Cell 1**: Install dependencies
   - **Cell 2**: Mount Google Drive and extract data
   - **Cell 3**: Imports and configuration
   - **Cell 4**: Load data and verify image paths
   - **Cell 5**: Load processor
   - **Cell 6**: Generate BLIP image captions (cached to Drive)
   - **Cell 7**: Prompt builder
   - **Cell 8**: Dataset and DataLoader
   - **Cell 9**: Load model and apply LoRA
   - **Cell 10**: Training loop with TTA inference
   - **Cell 11**: Generate submission.csv (Inference)

## Approach Summary
- Base model: `HuggingFaceTB/SmolVLM-500M-Instruct`
- Parameter-efficient fine-tuning: LoRA applied to attention and MLP layers (r=7, <5M trainable parameters)
- Image captioning: BLIP-generated captions injected into prompt
- Rich prompt: image caption + subject/grade/topic metadata + lecture + hint + question + choices
- Inference: logit-based scoring with Test-Time Augmentation (3 prompt variants)
