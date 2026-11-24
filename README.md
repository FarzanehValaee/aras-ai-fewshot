# aras-ai-fewshot-text2sql

Few-shot example construction and text/SQL conversion from the aras-ai-data service.
These are not four separate projects. They are one pipeline, which is why they live in one repo.

## Flow

1. Example construction — `data_service/services/fewshot_bootstrap.py` and `data_service/agent/icrl.py`
   build question/SQL pairs from the schema (templates or model generation) and store them after the SQL actually runs.
2. Text to SQL — `data_service/agent/pipeline.py`
   turns the question into a SELECT using those examples (`_sql_generation_prompt`, `_complete_sql`).
3. SQL to text — the same `pipeline.py`
   turns the query result into an English answer (`_grounding_prompt`, `answer_generate` stage).
4. Analysis — scripts in `benchmark/` plus `check_leakage.py` and `pool_forensics.py`
   measure example quality, difficulty, and dev-set leakage.

## Where to write new examples

- Fixed pairs: `data_service/services/fewshot_bootstrap.py`, function `_template_pairs`
- Model-generated pairs: `data_service/agent/icrl.py`, function `generate_pair`
- Easy / medium / hard quotas: `data_service/agent/fewshot_skeleton.py`, function `fewshot_quota`
- Previous sample output: `benchmark/fewshot_pool_clean.json`

## Run

This slice is still wired to the Django models and LLM router of that service.
Secrets and `.env` are not in the zip. For a local run, copy `.env.example`.
Keys are environment variables only: `ROUTER_API_KEY` or `OPENAI_API_KEY`.

```bash
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt
```

Experiment parameters live in `configs/repro.yaml`.
