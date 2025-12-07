# Hex20 Model Training Tool

This repository contains the standalone scripts used to generate synthetic telemetry data and train the AI models for the **Hex20 Satellite Dashboard** project.

## Purpose

The main `satelite-nightly-test-tool` relies on pre-trained Machine Learning models to detect anomalies. This toolchain allows you to:
1.  **Generate** synthetic training data (representing "normal" orbital behavior).
2.  **Train** the Isolation Forest and Linear Regression models.
3.  **Export** the `.joblib` model files for use in the main application.

## Files

*   `generator.py`: Core logic for simulating satellite physics (battery drain, solar charging, temperature cycles).
*   `generate_data.py`: Script to run the generator and save 24 hours of data to `data/telemetry.bin`.
*   `train.py`: Script that reads the binary data, trains the models, and saves them to `model/`.

## How to Run

### 1. Setup

Install the required dependencies:

```bash
pip install -r requirements.txt
```

### 2. Generate Data

Create the synthetic dataset. This will create a `data/` folder and a `telemetry.bin` file.

```bash
python generate_data.py
```

### 3. Train Models

Run the training script. This will read the generated data and output the trained models.

```bash
python train.py
```

Output:
*   `model/isoforest.joblib`: The Anomaly Detection model.
*   `model/scaler.joblib`: The feature scaler (required for the model).
*   `model/lr_battery.joblib`: A simple regression model for battery trends.

## Integration with Satellite Tool

To use these new models in the main **Hex20 Satellite Dashboard**:
1.  Copy the contents of the `model/` folder generated here.
2.  Paste them into `satelite nightly test tool/backend/model/`, overwriting the existing files.
3.  Restart the backend server.
