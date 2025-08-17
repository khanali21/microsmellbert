# MicroSmellBERT for Spring PetClinic Microservices

This repository contains scripts for detecting intra-service anti-patterns (Mega-Service, CRUDy Service, Ambiguous Service) in the Spring PetClinic microservices application using a hybrid dual-input deep learning model (metrics + CodeBERT embeddings). It's part of your MSc AI Capstone Project on code smell deduction and quality prediction for cloud-native development.

## Overview
- **Objective**: Validate the hybrid methodology from your thesis chapter on a new benchmark (PetClinic v2.6.7).
- **Key Components**:
  - Extraction: Copy Java files from repo.
  - Preprocessing: Clean Java code.
  - Labels: Heuristic generation with public dataset augmentation.
  - Metrics: Extract using CK tool.
  - Embeddings: Semantic features via CodeBERT.
  - Training: Dual NN model for multi-label classification.
  - Inference: Predict smells with confidence scores and visualizations.
- **Dataset**: ~7 services from https://github.com/spring-petclinic/spring-petclinic-microservices/tree/v2.6.7.

## Prerequisites
- **Environment**: Python 3.12.3 virtual env (updated from thesis setup for compatibility).
- **Libraries**: Install via pip: `torch transformers pandas numpy scikit-learn matplotlib javalang re shutil subprocess logging pickle`.
- **Tools**: CK JAR (`ck.jar`) in `../ck/` (download from https://github.com/mauricioaniche/ck).
- **Repo Clone**: `git clone -b v2.6.7 https://github.com/spring-petclinic/spring-petclinic-microservices.git petclinic-microservices`.
- **Configuration**: Update `config.json` with `'project': 'petclinic'`, `'repo_path'`: path to cloned repo, and other paths/thresholds as needed.
- **Public Datasets**: Place CSVs (e.g., from Figshare) in `public_datasets/` for label augmentation (optional).
- **Directories**: Scripts create `data/services`, `metrics`, `embeddings`, `models`, `results` automatically.

## Training and Inference Steps
Follow these steps in sequence. All scripts are in the project root, use `config.json` for paths/project name, and include logging for debugging. Outputs use `{project}` (e.g., `petclinic`) in filenames.

1. **Extract Java Files**:
   - Run: `python extract_java_files.py`
   - Output: .java files copied to `data/services/<service>/`.
   - Purpose: Organize source files per service; logs detected services and copied files.

2. **Preprocess Code**:
   - Run: `python preprocess_code.py`
   - Output: '.pre' files in `data/services/<service>/`.
   - Purpose: Clean code for embeddings (removes comments, normalizes vars); logs processed files per service.

3. **Generate Labels**:
   - Run: `python generate_labels.py`
   - Output: `data/labels_{project}.csv`.
   - Purpose: Heuristic labels; augments with public data if available; logs generated labels.

4. **Extract Metrics**:
   - Run: `python extract_metrics.py`
   - Output: `metrics/metrics_{project}.csv` (aggregated) and raw per-service in `metrics/<service>/`.
   - Purpose: Structural features via CK tool; logs extraction per service.

5. **Extract Embeddings**:
   - Run: `python extract_embeddings.py`
   - Output: `embeddings/embeddings_{project}.pkl` (DataFrame with service embeddings).
   - Purpose: Semantic features using CodeBERT; logs processing and shapes.

6. **Train Model**:
   - Run: `python train_model.py`
   - Output: `models/anti_pattern_model_{project}.pth`, `data/train_data_{project}.npz`, `data/test_data_{project}.npz`.
   - Purpose: Train dual NN; prints/logs loss and test metrics. 10 epochs, small batch for limited data.

7. **Infer Smells**:
   - Run: `python infer_smells.py`
   - Output: `results/predictions_{project}.csv`, `results/heatmap_predictions_{project}.png`, `results/training_loss_curve_{project}.png`.
   - Purpose: Predict probabilities/preds; flags high-conf issues; visualizes results; uses test split or fallback; logs data loading and results.

## Troubleshooting
- **Small Dataset**: PetClinic has only 7 services—expect potential overfitting; consider combining with train-ticket data.
- **Errors**:
  - CK.jar not found: Download and place correctly.
  - No .pre files: Ensure preprocessing ran after extraction.
  - Model dims mismatch: Verify metric_cols in config/train script match extract_metrics output.
  - Check logs: All scripts now use Python logging for detailed info/warnings/errors (e.g., file counts, shapes, exceptions).
- **Enhancements**: Logging added for better debugging; consider GPU support in torch for larger datasets.
- **Capstone Alignment**: This setup validates your hybrid model on a new benchmark, supporting objectives like feasibility and DevOps integration.

For questions or extensions (e.g., CI workflow), refer to your thesis chapter or consult supervisor.