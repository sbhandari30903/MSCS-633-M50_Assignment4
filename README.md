# AutoEncoder Fraud Detection — Assignment 4

**Overview:** This repository contains code for anomaly-based credit-card fraud detection using a PyOD AutoEncoder. The script downloads the anonymized Kaggle credit-card transactions dataset at runtime (via `kagglehub`) and evaluates an autoencoder as an outlier detector.

**Dataset:** The dataset is fetched automatically from Kaggle (`whenamancodes/fraud-detection`, file `creditcard.csv`) using `kagglehub`. You do not need to (and should not) commit the CSV to the repository — the script will download it when run. The dataset contains columns `Time`, `V1`..`V28`, `Amount`, `Class` (`Class = 1` indicates fraud, `0` indicates normal).

**Quick Start**

- **Install dependencies (PowerShell):**

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

- **Install dependencies (WSL / Linux / macOS):**

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

- **Run training & evaluation:**

```powershell
python src\train_autoencoder.py
```

Running the script will load `data/creditcard.csv` (or download it via `kagglehub`), preprocess features, train an AutoEncoder (trained primarily on normal transactions), and evaluate on a hold-out test set. The script saves two example plots to the repository root:

- `autoencoder_roc_curve.png`
- `autoencoder_score_distribution.png`

**Files**

- `src/train_autoencoder.py`: Main training / evaluation script. It downloads the dataset using `kagglehub`, performs preprocessing, trains a `pyod` AutoEncoder, evaluates using a tuned threshold, and saves plots.
- `requirements.txt`: Python dependencies required to run the code.

**Key Implementation Notes**

- The code treats fraud detection as an outlier detection problem and fits the AutoEncoder primarily on normal (`Class=0`) transactions.
- The AutoEncoder hyperparameters used in the script are reasonable starting values (see `src/train_autoencoder.py`). Tuning `hidden_neuron_list`, `epoch_num`, `batch_size`, `contamination`, and `lr` is recommended.
- A custom threshold is selected in the script (99.5% quantile on validation anomaly scores). Adjust this threshold to control precision/recall trade-offs.

**Reproducibility**

- The code sets `random_state=42` where applicable for reproducible splits and model randomness.
- Use the provided `requirements.txt` to pin dependencies. If you need a clean environment, recreate the virtualenv and reinstall.

**Typical Workflow**

1. (Optional) Place `creditcard.csv` under the `data/` directory, or ensure internet + Kaggle credentials for `kagglehub`.
2. Create & activate a virtual environment.
3. Install dependencies with `pip install -r requirements.txt`.
4. Run `python src/train_autoencoder.py` to train and evaluate.
