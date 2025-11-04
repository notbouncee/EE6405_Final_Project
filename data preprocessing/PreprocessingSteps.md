# Data Preprocessing

## Download the dataset

Download AMItheAsshole.sqlite dataset from https://www.kaggle.com/datasets/jianloongliew/reddit/data

## Quick setup and run (macOS / zsh)

These steps create an isolated virtual environment, install dependencies from `requirements.txt`, and run the preprocessing script `redditAITApreprocessing.py`.

#### 1. Put the dataset in the expected location

Place the downloaded SQLite file at `data/raw/AmItheAsshole.sqlite` (the scripts assume the DB lives under `data/raw`).

#### 2. Create and activate a virtual environment (zsh)

```bash
# create a venv in the repo root
python3 -m venv .venv

# activate it (zsh)
source .venv/bin/activate

# upgrade pip (optional but recommended)
python -m pip install --upgrade pip
```

#### 3. Install dependencies from `requirements.txt`

If you already have `requirements.txt` in the repo:

```bash
pip install -r requirements.txt
```

#### 4. Run the preprocessing script

The preprocessing script is in the folder `data preprocessing` (note the space). Two options to run it from the repo root:

Option A — call the script with a quoted path:

```bash
python "data preprocessing/redditAITApreprocessing.py"
```

Option B — change directory then run:

```bash
cd "data preprocessing"
python redditAITApreprocessing.py
cd -
```

5) When done

```bash
# deactivate the venv
deactivate
```

## Notes & troubleshooting
- If you get import errors, make sure you installed the packages listed in `requirements.txt` while the `.venv` is activated.
- The script expects `data/raw/AmItheAsshole.sqlite` to exist — if you placed it elsewhere, update the script path or move the file.
- If the script fails because a column or table is missing, open the SQLite file (e.g., `sqlite3 data/raw/AmItheAsshole.sqlite`) and inspect tables or run small `pandas.read_sql` queries to confirm schema.

#### Optional: create a reusable runner command in macOS zsh

Add the following small helper (one-liner) to run the script from repo root while automatically activating the venv:

```bash
(.venv/bin/activate && python "data preprocessing/redditAITApreprocessing.py")
```

This activates `.venv` for the duration of the command and then returns you to the same shell state.
