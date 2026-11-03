# Fruit Ripeness Ordinal Classification & Explanations

This repository implements ordinal regression for fruit ripeness classification using **CORAL** (COnsistent RAnk Logits; Cao et al., 2020) and a standard **Cross-Entropy Baseline** on top of a pretrained ResNet-18 backbone, with **Grad-CAM** visual interpretability.

---

## 1. Ripeness Classes & Stages

The model uses the ordered labels supplied by the selected dataset. The
primary mango/banana dataset has three stages: **Unripe → Ripe → Overripe**.
Other supported datasets may have a different number or naming of stages; the
checkpoint preserves the class names used during training.

---

## 2. Public Mango & Banana Ripeness Dataset

The primary in-the-wild dataset is the public
[Fruit-Ripeness-Dataset on Kaggle](https://www.kaggle.com/datasets/asadullahprl/fruits-ripeness-classification-dataset),
which includes mango and banana images labeled **Unripe**, **Ripe**, and
**Overripe**. The Kaggle metadata declares a CC0 public-domain license. Since
the dataset has three labels, the model uses three ordinal stages rather than
inventing a fourth.

Download and prepare the mango/banana subset while preserving the original
Train/Test split:
```powershell
python prepare_public_ripeness.py --target data_mango_banana
```
The script uses KaggleHub and installs it if needed. If Kaggle requests sign-in
or consent, complete that in your Kaggle account and rerun the command. The
selected subset contains about 2,600 training and 90 held-out test images;
validation is split from the training partition.

### Optional: Fruits-360

The project seamlessly supports the **Fruits-360 dataset** (Mihai Oltean) in two convenient ways:

### Option A: Using an Existing Local Fruits-360 Directory
If you have downloaded Fruits-360 (from Kaggle or Mendeley Data):

1. **In-place Direct Training** (no copying needed):
   ```bash
   python train.py --data-root "path/to/fruits-360/Training" --variant ordinal
   ```
   `dataset.py` automatically detects the Fruits-360 directory structure and maps the classes on the fly.

2. **Organize into `data/` using `prepare_fruits360.py`**:
   ```bash
   # Tomato Ripeness (Tomato not Ripened -> Yellow -> Cherry Red -> Maroon)
   python prepare_fruits360.py --source "path/to/fruits-360" --preset tomato

   # Apple Ripeness (Granny Smith -> Golden 1 -> Golden 2 -> Red 1)
   python prepare_fruits360.py --source "path/to/fruits-360" --preset apple

   # Multi-Fruit Ripeness (Tomatoes, Apples, Bananas, Avocados)
   python prepare_fruits360.py --source "path/to/fruits-360" --preset mixed
   ```

### Option B: Auto-Download Real Fruits-360 Samples from GitHub
If you don't have Fruits-360 locally, you can automatically download authentic Fruits-360 images directly from the official GitHub mirror:
```bash
python prepare_fruits360.py --download --preset tomato --samples-per-class 50
```

For the full Fruits-360 training set, download and prepare the mapped tomato
classes into a separate folder (so existing `data/` images are left untouched):
```bash
python download_dataset.py --preset tomato --target data_fruits360
```

---

## 3. Training the Models

Images are cropped around foreground pixels before training, evaluation, and
web inference. The crop helper keeps the original image when it cannot identify
a distinct foreground.

Install the requirements in your environment (select a CUDA-enabled PyTorch
build from the official PyTorch instructions if you plan to train on NVIDIA
GPU), then prepare the dataset and train the CORAL model:

```powershell
python -m pip install -r requirements.txt
python prepare_public_ripeness.py --target data_mango_banana
python train.py --variant ordinal --data-root data_mango_banana --device cuda --batch-size 8 --epochs 30
python evaluate.py --variant ordinal --data-root data_mango_banana --device cuda --no-gradcam
```

CUDA training automatically enables mixed precision. Reduce `--batch-size` if
the GPU runs out of memory; use `--device cpu` for CPU training.

Key features:
* **Source test split preserved**; stratified validation comes from Train only.
* **Class-balanced sampling** for training batches.
* **Strictly ordered CORAL thresholds** for consistent cumulative probabilities.
* **AdamW** optimizer with **ReduceLROnPlateau** scheduler monitoring validation MAE.
* **Early stopping** with patience = 10 epochs.
* Best model checkpoint saved to `checkpoints/best_ordinal.pth`.
* Epoch metrics logged to `logs/training_ordinal.csv`.

The included `.gitignore` intentionally excludes downloaded images, virtual
environments, generated reports, and model checkpoints. Prepare the dataset
and train a checkpoint locally before running inference.

To start the web app after training:

```powershell
python app.py
```

The dashboard is available at `http://127.0.0.1:5000`.

---

## 4. Evaluation & Grad-CAM Visualizations

Evaluate the trained checkpoint on the Kaggle-provided test split:

```powershell
python evaluate.py --variant ordinal --data-root data_mango_banana --device cuda --no-gradcam
```

Outputs:
* **Metrics**: Top-1 Accuracy, MAE, Off-by-1 Accuracy, and Per-Class MAE.
* **Confusion Matrix** for the selected model.
* **Evaluation CSV**: `logs/evaluation_results.csv`.
* **Grad-CAM Heatmaps**: Generated under `gradcam_outputs/` showing whether the models focus on ripeness-relevant cues (color transitions, surface texture) rather than background or orientation.

The current trained model scored 97.75% accuracy (87/89 images) on the
dataset's held-out test split. This split is small, so the result should not
be treated as a guarantee of performance on new cameras, fruit varieties, or
lighting conditions.
