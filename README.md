# Job Application Portal (Streamlit)

This is a simple Streamlit-based job application portal where HR can upload job descriptions and applicants can upload resumes. Uploaded resumes are scored against the job description and scoring results are written to a `scores/<job_id>/scoring_results.json` file.

## Install

1. Create a Python virtual environment (recommended):

```powershell
py -m venv .venv
.\.venv\Scripts\Activate.ps1
```

2. Install dependencies:

```powershell
py -m pip install -r requirements.txt
```

## OpenAI API Key

The app uses OpenAI for some formatting and scoring helpers. You can provide your OpenAI API key either by editing a local `config.py` (not recommended for source control) or via an environment variable.

Option A — `config.py` (quick local dev):

Create a file named `config.py` at the project root with the following content:

```python
# config.py (LOCAL ONLY — do NOT commit to git)
OPENAI_API_KEY = "sk-REPLACE_WITH_YOUR_KEY"
```

Make sure `config.py` is ignored by Git (it is included in `.gitignore`).

Option B — Environment variable (recommended):

Set the environment variable `OPENAI_API_KEY` in your shell / CI environment. For PowerShell:

```powershell
$env:OPENAI_API_KEY = 'sk-REPLACE_WITH_YOUR_KEY'
# Or for persistent storage, add it to your user environment variables via Windows settings
```

The code will try to import the key from `config.py` and otherwise you can modify the code to read from `os.environ['OPENAI_API_KEY']`.

## Run

```powershell
# from project root
py -m streamlit run app.py
```

## Tests

Run the provided pytest to validate the migration utility:

```powershell
py -m pytest -q tests/test_migration_and_marker.py
```

## Where scoring files are stored

Scoring JSONs and logs are stored under the `scores/` directory at the project root, with one subdirectory per job (for example `scores/job_1/`). Each job folder contains `scoring_results.json`, `score_log.txt`, and `last_score_location.txt` (the latter is a small marker pointing to the JSON location).

## Security

- Never commit `config.py` or your secret API keys to source control. Use `.env` or environment variables in production.
- Rotate API keys if you think they were exposed.

## Troubleshooting

- If you don't see scoring results, check `scores/<job_id>/score_log.txt` and `scores/<job_id>/last_score_location.txt` for clues. If the migration runs, the HR sidebar should also show the migration summary when you load the app.
