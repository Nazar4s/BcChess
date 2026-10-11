# Chess AI research project

Bachelor's project exploring modular neural networks for chess position evaluation, explainability, and human style adaptation.

## Repository contents

- `Bc.Job/` — literature notes, research concepts, and project references.
- `dataToNpy.py` — converts PGN games into board tensors, move labels, and game results.
- `data_read.py` — current CNN training prototype.
- `selen.py` — Selenium/Chessboard UI inference prototype.

Large PGN files, generated NumPy datasets, and model checkpoints are excluded from Git. Put local data in `chessdata/`, generated arrays in `processed_data/`, and checkpoints in `models/`.

## Current status

The scripts are research prototypes and need fixes before the training and browser inference flows run end to end. In particular, the training loop currently has output/target handling errors. See the project notes in `Bc.Job/` for the research direction.

## Setup

Install Python and the dependencies used by the scripts:

```powershell
python -m pip install numpy python-chess torch torchvision scikit-learn selenium
```

Then place a PGN at `chessdata/lichess_elite_2020-01.pgn` and preprocess it:

```powershell
python dataToNpy.py
```

The preprocessing step can produce several gigabytes of data. Training and Selenium inference are not yet reliable until their current implementation issues are fixed.
