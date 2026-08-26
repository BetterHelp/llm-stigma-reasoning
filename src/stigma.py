"""
Used to evaluate models in the stigma experiments.

"""

import argparse
import datetime
import logging
import os

import numpy as np
import pandas as pd
from tqdm import tqdm

from unified_client import create_client

from generate_stigma_data import UNKNOWN

from utils import generate_short_folder_name, extract_blocks_to_dict, prompt_to_messages, CLAUDE_REINFORCEMENT_PROMPT, COT_SYSTEM_PROMPT, NTCOT_SYSTEM_PROMPT

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS_DIR = os.path.join(REPO_ROOT, "results")
PROMPTS_FILE = os.path.join(REPO_ROOT, "data", "prompts.jsonl")
OPTIONS = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)
logger.setLevel(logging.INFO)


def get_option(text: str) -> str:
    """Extract the option letter from raw model output, e.g. '(A)' -> 'A'."""
    token = text.strip()
    token = token.replace("▁", "").replace("(", "").replace(")", "").replace(":", "").replace(".", "").replace("*", "")
    return token.split()[0]


def find_answer(option: str, answers: list[str]) -> str:
    """Map an option letter to the corresponding answer string."""
    option = option.upper()
    valid = OPTIONS[:len(answers)].upper()
    if option in valid:
        return answers[OPTIONS.index(option)]
    logger.warning("No option included")
    return None


def get_answer(row):
    text = row["output"]
    answers = row["options"]
    if not text:
        logger.error(f"No response for row: {row}")
        return None
    option = get_option(text)
    answer = find_answer(option, answers)
    return answer


def code_answer(row):
    """
    np.nan if "Do not know" or answer missing
    0 if "Not at all likely" or "Not very likely", 1 otherwise
    """
    answer = row["answer"]
    answers = row["options"]
    if not answer or answer == UNKNOWN or answer not in answers:
        code = np.nan
    else:
        answer_pos = answers.index(answer)
        code = answer_pos not in [0, 1]
    return code


def main():
    parser = argparse.ArgumentParser(prog="evaluate")
    parser.add_argument(
        "--output-directory",
        default=RESULTS_DIR,
        help="Where to output the results.",
    )
    parser.add_argument("--model", required=True, help="The model to query.")
    parser.add_argument(
        "--source", choices=["bedrock"], default="bedrock", help="which source to query"
    )
    parser.add_argument(
        "--use-cot",
        action="store_true",
        default=False,
        help="Whether to use a chain-of-thought prompt",
    )
    parser.add_argument(
        "--ntcot",
        action="store_true",
        default=False,
        help="Use non-therapist COT mode (generic respondent framing, implies --use-cot)",
    )
    parser.add_argument(
        "--prompts-file",
        default=PROMPTS_FILE,
        help="Path to the prompts JSONL file (default: data/prompts.jsonl)",
    )
    parser.add_argument(
        "--demo-mode",
        action="store_true",
        default=False,
        help="Whether to use demo mode (debug and log all outputs and inputs for a single prompt)",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Only process the first N prompts (for incremental runs)",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=1,
        help="Number of concurrent workers for parallel prompting",
    )
    args = parser.parse_args()

    df = pd.read_json(args.prompts_file, lines=True)

    if args.limit:
        df = df.head(args.limit)

    use_cot = args.use_cot or args.ntcot
    cot_prompt = NTCOT_SYSTEM_PROMPT if args.ntcot else COT_SYSTEM_PROMPT

    if 'claude' in args.model and not use_cot:
        reinforcement_prompt = CLAUDE_REINFORCEMENT_PROMPT
    else:
        reinforcement_prompt = None

    source = args.source
    model = args.model

    # Opus 4.5 plain mode: extra few-shot + assistant prefill to force single-letter answers
    is_opus_plain = 'opus' in model.lower() and not use_cot

    # --- Resume logic: compute output path early and filter already-completed prompts ---
    run_dir = os.path.join(
        args.output_directory,
        generate_short_folder_name(model, use_cot=use_cot, ntcot=args.ntcot),
    )
    now = datetime.datetime.now().date().isoformat()
    file_path = os.path.join(run_dir, f"{now}.jsonl")

    existing_df = None
    if os.path.exists(file_path):
        existing_df = pd.read_json(file_path, lines=True)
        existing_prompts = set(existing_df["prompt"].tolist())
        logger.info(f"Found {len(existing_prompts)} existing results in {file_path}, skipping those")
        df = df[~df["prompt"].isin(existing_prompts)]
        if df.empty:
            print(f"All prompts already completed in {file_path}")
            return

    prompts = {
        prompt: prompt_to_messages(prompt, use_cot=use_cot, cot_prompt=cot_prompt, reinforcement_prompt=reinforcement_prompt, opus_plain_mode=is_opus_plain)
        for prompt in df["prompt"]
    }

    texts = []

    max_tokens = 50
    if use_cot:
        max_tokens = 4096

    with create_client(
        source=source,
        model=model,
        max_tokens=max_tokens,
        temperature=0,
        max_parallel=args.workers,
        concurrent=args.workers > 1,
    ) as client:
        if args.demo_mode:
            # pick a random prompt and use it for the entire loop
            df = df.sample(1)
            prompts = {df["prompt"].iloc[0]: prompts[df["prompt"].iloc[0]]}
            total_prompts = 1
            print(f"Processing {total_prompts} prompt...")
            print("="*80)
            print(f"Input Prompt: \n\n {prompts[df['prompt'].iloc[0]]}")
            print("="*80)
            
        outputs = client(keys_to_messages=prompts)

        # Retry failed prompts (up to 2 more attempts)
        max_retries = 2
        for attempt in range(1, max_retries + 1):
            failed_keys = {
                k: prompts[k] for k, v in outputs.items()
                if v.get("text") is None or v.get("text", "").strip() == ""
            }
            # Also include prompts missing from outputs entirely
            for k in prompts:
                if k not in outputs:
                    failed_keys[k] = prompts[k]

            if not failed_keys:
                break
            logger.info(f"Retry attempt {attempt}/{max_retries}: re-querying {len(failed_keys)} failed prompts")
            retry_outputs = client(keys_to_messages=failed_keys)
            for k, v in retry_outputs.items():
                if v.get("text") is not None and v.get("text", "").strip() != "":
                    outputs[k] = v

        # Log final failure count
        final_failed = sum(
            1 for k in prompts
            if k not in outputs or outputs[k].get("text") is None or outputs[k].get("text", "").strip() == ""
        )
        if final_failed > 0:
            logger.warning(f"{final_failed} prompts still failed after {max_retries} retries")

        for prompt in tqdm(df["prompt"], desc="Processing responses"):
            text = None
            if prompt in outputs:
                text = outputs[prompt]["text"]
                # Opus 4.5 plain mode: prepend the prefill so <final>...</final> tag is complete
                if is_opus_plain and text:
                    text = "<final>" + text

            if args.demo_mode:
                print("="*80)
                print(f"Response: {text}")
                print("="*80)

            # process final response as json
            final_response = extract_blocks_to_dict(text)
            rationale = final_response["rationale"] if "rationale" in final_response else ""
            texts.append({"output": final_response["response"],
                        "prompt": prompt,
                        "reasoning": rationale,
                        "raw_response": text
                        })

    if args.demo_mode:
        return
    result = pd.merge(df, pd.DataFrame(texts), on="prompt")
    result["answer"] = result.apply(get_answer, axis=1)
    result["code"] = result[["answer", "options"]].apply(code_answer, axis=1)

    # Concatenate with existing results if resuming
    if existing_df is not None:
        result = pd.concat([existing_df, result], ignore_index=True)

    if not os.path.exists(run_dir):
        os.makedirs(run_dir)

    print(f"Outputting to {file_path}")

    result.to_json(file_path, lines=True, orient="records", index=False)


if __name__ == "__main__":
    main()
