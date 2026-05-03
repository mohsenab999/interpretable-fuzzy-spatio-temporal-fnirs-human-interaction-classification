# fNIRS Human Interaction Classification With Interpretable Fuzzy Filters

This project is a PyTorch implementation for classifying human interaction states from dyadic functional near-infrared spectroscopy (fNIRS) data. The implementation is built around fuzzy neural modeling so that the trained model can provide both prediction results and interpretable spatial/temporal fuzzy firing patterns.

## Research Foundation

This code is based on two uploaded papers:

1. 
   - Paper: "A Fuzzy Logic-Based Approach to Predict Human Interaction by Functional Near-Infrared Spectroscopy"
   - DOI: `10.1109/TFUZZ.2025.3528376`
   - Role: provides the main fNIRS human-interaction prediction idea and task setting.

2. 
   - Paper: "iFuzzyTL: Interpretable Fuzzy Transfer Learning for Steady-State Visual Evoked Potentials Brain-Computer Interfaces System"
   - DOI: `10.1109/TSMC.2025.3614244`
   - Role: provides the interpretable fuzzy spatial and temporal filter idea used to build the model architecture.

In this project, the fNIRS interaction-classification problem from the first paper is implemented using spatial and temporal fuzzy filtering concepts inspired by the second paper.

## Project Goal

The model predicts two interaction conditions:

- Class `0`: no-touch condition from `block1`
- Class `1`: hand-holding condition from `block2`

For each dyad pair, the project extracts Subject 1 and Subject 2 fNIRS trials, concatenates them into one dyadic tensor, trains an interval type-2 fuzzy spatio-temporal neural network, and then visualizes the learned fuzzy firing strengths.

## Model Design

The main model is defined in `IFT2_Spatio_Temporal_BCI_Network.py`.

The network contains:

- A spatial interval type-2 fuzzy filter in `torch_spatial_IT2_fuzzy_filter.py`
- A temporal interval type-2 fuzzy filter in `torch_temporal_IT2_fuzzy_filter.py`
- A lightweight PyTorch classifier for binary prediction
- Post-training extraction of fuzzy centers, projection matrices, and firing strengths

The spatial fuzzy filter learns rule-based channel patterns, while the temporal fuzzy filter learns rule-based time-window patterns. These learned firing strengths are saved and plotted to support interpretability.

## Main Files

- `train_pytorch_model.py`: extracts data, trains the model, evaluates accuracy, and saves fuzzy outputs.
- `IFT2_Spatio_Temporal_BCI_Network.py`: interval type-2 spatio-temporal fuzzy BCI network.
- `torch_spatial_IT2_fuzzy_filter.py`: spatial fuzzy filter layer.
- `torch_temporal_IT2_fuzzy_filter.py`: temporal fuzzy filter layer.
- `your_extraction_script.py`: fNIRS loading and preprocessing logic.
- `fuzzy_model_result_plot.py`: spatial fuzzy firing heatmaps.
- `fuzzy_temporal_result_plot.py`: temporal fuzzy firing heatmap.
- `center_recovery.py`: helper for recovering fuzzy centers.

## Data Layout

The training script expects pair folders in this form:

```text
dataset/fnirs_raw_data/<pair_id>/block1/Subject1/
dataset/fnirs_raw_data/<pair_id>/block1/Subject2/
dataset/fnirs_raw_data/<pair_id>/block2/Subject1/
dataset/fnirs_raw_data/<pair_id>/block2/Subject2/
```

The included `dataset/pkl/` folder contains extracted pickle files for many pairs.

## Dataset Download

The dataset can be downloaded from the OSF/PsyArXiv page:

```text
https://osf.io/preprints/psyarxiv/a27ns_v1
```

After downloading and extracting the dataset, place the raw fNIRS files under:

```text
dataset/fnirs_raw_data/
```

The training script expects each pair to contain `block1` and `block2` folders, with `Subject1` and `Subject2` subfolders inside each block.

## Requirements

Install the required packages:

```bash
pip install numpy torch scikit-learn matplotlib mne
```

On Apple Silicon, the training script automatically uses the `mps` device when available. Otherwise, it uses CPU.

## Training

Run:

```bash
python train_pytorch_model.py
```

The script performs these steps:

1. Loads fNIRS trials for all listed dyad pairs.
2. Concatenates Subject 1 and Subject 2 data.
3. Builds labels for no-touch and hand-holding conditions.
4. Splits the dataset into training and testing sets.
5. Trains the interval type-2 fuzzy spatio-temporal network.
6. Prints final test accuracy.
7. Saves fuzzy centers, projection matrices, and firing strengths.


## Plotting

After training, plot the spatial fuzzy results:

```bash
python fuzzy_model_result_plot.py
```

This creates heatmaps for Subject 1 HbO and HbR spatial firing strengths.

Then plot the temporal fuzzy results:

```bash
python fuzzy_temporal_result_plot.py
```

This creates a temporal fuzzy firing-strength heatmap.

## Typical Workflow

```bash
python train_pytorch_model.py
python fuzzy_model_result_plot.py
python fuzzy_temporal_result_plot.py
```

