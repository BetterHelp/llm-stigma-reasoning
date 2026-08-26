# Analyzing LLM Reasoning to Uncover Mental Health Stigma

Code and data for the paper [Analyzing LLM Reasoning to Uncover Mental Health Stigma](https://arxiv.org/abs/2604.25053), accepted to EMNLP 2026 (main conference).

The paper studies stigma toward people with mental health conditions in large language models. Instead of only scoring multiple-choice answers, we analyze the intermediate reasoning of LLMs to find stigmatizing language and the rationales behind it. A clinician-informed taxonomy is used to tag stigma patterns in the reasoning and rate their severity.

![Examples of stigma patterns found in LLM reasoning](figures/Intro_paper_main.png)

The figure shows three examples of what the analysis uncovers. A model reads a vignette about a fictional person, answers a social distance question, and its reasoning reveals contradictory logic, stigmatizing logic, or overt stigma that the multiple-choice answer alone would hide.

## Pipeline

The experiments run in four steps. All model calls go through AWS Bedrock.

1. `src/generate_stigma_data.py` builds vignettes and questions and writes prompt files to `data/`.
2. `src/stigma.py` queries a model with the prompts and writes answers to `results/`.
3. `src/tag_results.py` tags the chain-of-thought reasoning in `results/` with the stigma taxonomy, using Claude on Bedrock as the judge.
4. `src/make_plots.py` computes stigma scores and writes the paper figures to `figures/`.

## Setup

Requires Python 3.11 or newer and an AWS account with Bedrock access to the models listed below.

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Create a `.env` file in the repo root with your AWS profile name:

```
AWS_PROFILE_NAME=your-profile
```

The Bedrock region defaults to `us-west-2`. Set `AWS_DEFAULT_REGION` to change it.

## Usage

Generate the prompt data (the files are already included in `data/`):

```bash
python src/generate_stigma_data.py
```

Run one experiment (one model, one prompting mode):

```bash
python src/stigma.py --model "us.anthropic.claude-sonnet-4-20250514-v1:0" --prompts-file data/prompts_reduced.jsonl --use-cot --workers 4
```

Modes: no flag (plain multiple choice), `--use-cot` (chain of thought), `--ntcot` (chain of thought with a generic respondent framing instead of a therapist framing).

Run all 24 experiments (8 models x 3 modes):

```bash
bash run_experiments.sh --workers 4
```

Tag the reasoning in all COT and NTCOT results (supports resume, safe to re-run):

```bash
python src/tag_results.py --workers 3
```

Generate the figures:

```bash
python src/make_plots.py
```

## Models

| Bedrock model ID | Short name |
|---|---|
| us.anthropic.claude-opus-4-5-20251101-v1:0 | claude-opus-45 |
| us.anthropic.claude-sonnet-4-20250514-v1:0 | claude-sonnet-4 |
| us.meta.llama3-3-70b-instruct-v1:0 | llama3.3-70b |
| deepseek.v3-v1:0 | deepseek-v31 |
| openai.gpt-oss-120b-1:0 | gpt-oss-120b |
| openai.gpt-oss-20b-1:0 | gpt-oss-20b |
| us.meta.llama4-maverick-17b-instruct-v1:0 | llama4-maverick-17b |
| us.meta.llama4-scout-17b-instruct-v1:0 | llama4-scout-17b |

Results are stored in `results/{short_name}`, `results/{short_name}_COT`, and `results/{short_name}_NTCOT`. Tagged reasoning is stored in `results/{short_name}_COT_Tagged` and `results/{short_name}_NTCOT_Tagged`. The tagging model is Claude Opus 4.5 on Bedrock (`--model sonnet` switches to Claude Sonnet 4).

## Repository layout

```
data/                         Prompts, vignettes, questions, taxonomy, human annotations
results/                      Model answers and tagged reasoning used in the paper
figures/                      Paper figures, produced by make_plots.py
run_experiments.sh            Runs all models and modes
src/
  generate_stigma_data.py     Builds vignettes, questions, and prompt files
  stigma.py                   Runs one model on the prompts
  tag_results.py              Tags reasoning with the stigma taxonomy
  make_plots.py               Computes scores and draws the figures
  unified_client.py           AWS Bedrock client for all model families
  prompts.py                  System prompts and few-shot examples
  utils.py                    Model registry and parsing helpers
  stigma_rubric.py            Maps survey questions to stigma interpretations
```

## Data

- `data/vignettes.jsonl`: 144 vignettes (8 conditions x 3 education levels x 3 races x 2 genders). The vignette templates come from the supplemental material of [Pescosolido et al. (2021)](https://pmc.ncbi.nlm.nih.gov/articles/PMC8693212/) and were extended with four additional conditions.
- `data/questions.jsonl`: 14 survey questions (illness attribution, causal attribution, social distance, and violence).
- `data/prompts.jsonl`: full cross of vignettes and questions (2016 prompts).
- `data/prompts_reduced.jsonl`: reduced set used for the paper experiments (336 prompts, one random race and gender per condition and education level, seed 42).
- `data/taxonomy.json`: the stigma taxonomy with 6 categories and few-shot examples for the tagging model.
- `data/pre-tagged-examples.csv`: tagged examples rated by human annotators, used to validate the tagging model.

All vignettes are synthetic and describe fictional people.

## Citation

```bibtex
@inproceedings{sankar2026analyzing,
  title={Analyzing LLM Reasoning to Uncover Mental Health Stigma},
  author={Sankar, Sreehari and Nafar, Aliakbar and Barman, Mona and Heitz, Hannah K. and Kumar, Ashwin and Tohidi, Pouria and Li, Dailun and Hussain, Danish and DuBois, Russell and Hasheminia, Hamed and Majzoubi, Farshad},
  booktitle={Proceedings of the 2026 Conference on Empirical Methods in Natural Language Processing (EMNLP)},
  year={2026}
}
```

## License

MIT. See [LICENSE](LICENSE).
