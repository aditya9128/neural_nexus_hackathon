# Chest X-Ray Multi-Label Classification

An end-to-end deep learning project for classifying 14 thoracic conditions from chest X-ray images using the NIH ChestXray14 dataset. The pipeline is built around a DenseNet121 backbone and focuses on practical training workflows such as memory-aware preprocessing, mixed-precision training, checkpointing, and reproducible evaluation.

## Project Highlights

- Multi-label classification for 14 thoracic findings
- DenseNet121-based model inspired by CheXNet-style setups
- Dask-powered metadata processing for large CSV handling
- Parquet caching for faster repeat experiments
- Mixed-precision training and gradient accumulation
- Patient-aware train/validation/test splitting
- Support for both evaluation and single-image inference

## Repository Layout

```text
chest-X-ray/
|-- app/
|-- backend/
|-- frontend/
|-- nih_chestxray_pipeline.py
|-- data.py
|-- graphs.py
|-- Dockerfile
|-- docker-compose.yml
|-- requirements.txt
|-- README.md
|-- README_Summary.md
`-- blueprint.txt
```

## Pipeline Overview

The core training workflow is centered in `nih_chestxray_pipeline.py`:

1. Discover image files from the NIH dataset structure.
2. Read and normalize metadata from CSV files.
3. Encode disease labels into a 14-class multi-label target vector.
4. Cache processed metadata to Parquet for faster reruns.
5. Build patient-safe train, validation, and test splits.
6. Train a DenseNet121 classifier with weighted BCE loss.
7. Evaluate AUROC and F1 on the held-out test set.
8. Run batch or single-image inference using the saved checkpoint.

## Model Summary

- Backbone: `DenseNet121`
- Task type: multi-label image classification
- Number of labels: `14`
- Default image size: `224 x 224`
- Loss: `BCEWithLogitsLoss` with class weighting
- Training optimizations: AMP, gradient accumulation, gradient clipping

## Dataset

This project expects the NIH ChestXray14 dataset and related split files:

- `Data_Entry_2017_v2020.csv`
- `train_val_list.txt`
- `test_list.txt`
- image folders such as `images_001/images/`, `images_002/images/`, and so on

If the official split files are unavailable, the pipeline can fall back to a random split strategy.

## Local Setup

```bash
pip install -r requirements.txt
python nih_chestxray_pipeline.py
```

If you plan to use Docker:

```bash
docker compose up --build
```

## Outputs

Typical generated artifacts include:

- `metadata.parquet`
- `best_model.pth`
- `test_predictions.csv`
- `training_curves.png`

## Notes

- The project is designed for large-scale chest X-ray experimentation, so storage and memory planning matter.
- GPU acceleration is strongly recommended for training.
- You may need to adjust dataset paths depending on whether you run locally, in Docker, or in Kaggle.
