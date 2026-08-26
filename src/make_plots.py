"""
Generate the paper figures from tagged experiment results.

Reads results/{model}[_COT|_NTCOT][_Tagged]/*.jsonl and writes PNG figures
to the output directory (default: figures/).

Usage:
    python make_plots.py
    python make_plots.py --results-dir results --output-dir figures
"""

import argparse
import glob
import os
import re
from typing import Dict, Optional

import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd

from stigma_rubric import (
    STIGMA_POSITIVE_QUESTIONS,
    STIGMA_NEGATIVE_QUESTIONS,
    evaluate_stigma_score,
)

# =============================================================================
# Config
# =============================================================================

# Models analyzed in the paper figures (short names, matching results/ folders)
MAIN_MODELS = [
    "deepseek-v31",
    "claude-sonnet-4",
    "claude-opus-45",
    "llama3.3-70b",
]

# Variant folder suffixes
VARIANTS = {
    "baseline": "",
    "cot": "_COT",
    "ntcot": "_NTCOT",
}

MODEL_COLORS = {
    "claude-sonnet-4": "#225240",
    "claude-opus-45": "#96e092",
    "llama3.3-70b": "#a1d9d8",
    "deepseek-v31": "#885942",
    "gpt-oss-120b": "#fcb13a",
    "gpt-oss-20b": "#ec7967",
    "llama4-scout-17b": "#577495",
    "llama4-maverick-17b": "#b0527c",
}

MODEL_COLOR_CYCLE = ["#225240", "#96e092", "#397a4a", "#a1d9d8", "#885942", "#FFBB28", "#8394CE"]

PRETTY_MODEL_NAMES = {
    "claude-opus-45": "Claude Opus 4.5",
    "claude-sonnet-4": "Claude Sonnet 4",
    "llama3.3-70b": "LLaMA 3.3 70B",
    "deepseek-v31": "DeepSeek v3.1",
}

ALL_CONDITIONS = [
    "borderline personality disorder", "eating disorder", "bipolar disorder", "psychosis",
    "depression", "schizophrenia", "alcohol dependence", "daily troubles",
]

TAG_CATEGORY_LABELS = [
    "Dangerousness or Threat",
    "Incompetence or Incapacity",
    "Pathologization of Normal or \n Cultural Behaviour",
    "Social Undesirability & Otherness",
    "Burden or Drain on Resources",
    "Treatment Stigma",
]

matplotlib.rcParams["hatch.linewidth"] = 0.8
plt.rcParams["figure.facecolor"] = "white"
plt.rcParams["axes.facecolor"] = "white"


def prettify_model_name(raw_name: str) -> str:
    """Pretty print a model or model_variant name."""
    base = raw_name
    if base.endswith("_COT"):
        base = base[:-4]
    if base in PRETTY_MODEL_NAMES:
        return PRETTY_MODEL_NAMES[base]
    for k, v in PRETTY_MODEL_NAMES.items():
        if k in base:
            return v
    return base.replace("-", " ").title()


def format_condition_label(cond: str) -> str:
    """Capitalize each word of a condition and put each on a new line."""
    return "\n".join(w.capitalize() for w in cond.split())


# =============================================================================
# Data loading
# =============================================================================

def find_latest_file(folder_path: str) -> Optional[str]:
    """Find the most recent .jsonl file in a folder based on filename date."""
    if not os.path.exists(folder_path):
        return None

    jsonl_files = glob.glob(os.path.join(folder_path, "*.jsonl"))
    if not jsonl_files:
        return None

    if len(jsonl_files) > 1:
        print(f"  Warning: Multiple files found in {folder_path}, using latest by filename")
        jsonl_files.sort()

    return jsonl_files[-1]


def load_model_data(results_dir: str, model_short_name: str, variant_suffix: str) -> Optional[pd.DataFrame]:
    """Load data for a specific model and variant."""
    folder_name = f"{model_short_name}{variant_suffix}"
    folder_path = os.path.join(results_dir, folder_name)

    file_path = find_latest_file(folder_path)
    if file_path is None:
        print(f"  No data found for {folder_name}")
        return None

    print(f"  Loading: {file_path}")

    # Remove "_Tagged" from variant suffix if present
    if variant_suffix.rsplit("_", 1)[-1] == "Tagged":
        variant_suffix = variant_suffix.rsplit("_", 1)[0]

    df = pd.read_json(file_path, lines=True)
    df["model"] = model_short_name
    df["variant"] = variant_suffix
    df["model_variant"] = model_short_name + variant_suffix
    df["tagged"] = 1 if variant_suffix in ("_COT", "_NTCOT") else 0
    return df


def load_all_data(results_dir: str) -> pd.DataFrame:
    """Load all data for the main models and variants."""
    print("Loading data...")
    all_dfs = []

    for model in MAIN_MODELS:
        for variant_suffix in VARIANTS.values():
            if variant_suffix in ("_COT", "_NTCOT"):
                df = load_model_data(results_dir, model, variant_suffix + "_Tagged")
            else:
                df = load_model_data(results_dir, model, variant_suffix)
                # Create columns with same names as tagged data to allow concatenation
                if df is not None:
                    for col_name in ["tags", "analysis_summary", "has_stigma", "how_many_stigma", "stigma_score"]:
                        df[col_name] = pd.NA
            if df is not None:
                all_dfs.append(df)

    if not all_dfs:
        raise ValueError(f"No data files found in {results_dir}!")

    combined_df = pd.concat(all_dfs, ignore_index=True)
    combined_df.rename(columns={"stigma_score": "stigma_severity"}, inplace=True)
    print(f"Loaded {len(combined_df)} total rows from {len(all_dfs)} model variants")
    return combined_df


# =============================================================================
# Score computation
# =============================================================================

def compute_stigma_scores(df: pd.DataFrame):
    """Compute stigma scores for each model/variant/condition combination."""
    results = {}

    # Filter to only stigma questions
    stigma_questions = STIGMA_POSITIVE_QUESTIONS + STIGMA_NEGATIVE_QUESTIONS
    stigma_df = df[df["question_only"].isin(stigma_questions)].copy()

    if stigma_df.empty:
        raise ValueError("No stigma questions found in data")

    # Add stigma evaluation for each row
    stigma_scores = []
    for _, row in stigma_df.iterrows():
        if pd.isna(row["answer"]) or row["answer"] is None:
            stigma_scores.append(None)
        else:
            eval_result = evaluate_stigma_score(row["question_only"], row["answer"], row["options"])
            stigma_scores.append(eval_result["is_stigmatizing"])

    stigma_df["is_stigmatizing"] = stigma_scores

    # Compute scores by model/variant/condition
    for model_variant in stigma_df["model_variant"].unique():
        mv_data = stigma_df[stigma_df["model_variant"] == model_variant]

        results[model_variant] = {}

        # All conditions are included (incl. the daily troubles control)
        if mv_data["is_stigmatizing"].isna().sum() > 0:
            print(f"Warning: NaN found in 'is_stigmatizing' for model variant: {model_variant}")

        valid_overall = mv_data[mv_data["is_stigmatizing"].notna()]
        results[model_variant]["overall"] = (
            valid_overall["is_stigmatizing"].mean() if len(valid_overall) > 0 else np.nan
        )

        # Reasoning-analysis stigma rate (from the LLM-judge tags)
        if model_variant.rsplit("_", 1)[-1] in ("COT", "NTCOT"):
            if mv_data["has_stigma"].isna().sum() > 0:
                raise ValueError(f"NaN found in 'has_stigma' for model variant: {model_variant}")
            results[model_variant]["reasoning_analysis"] = (
                mv_data["has_stigma"].apply(lambda x: 1 if x == "Yes" else 0 if x == "No" else np.nan).mean()
            )

        # Per-condition scores
        for condition in ALL_CONDITIONS:
            condition_data = mv_data[mv_data["condition"] == condition]
            valid_condition = condition_data[condition_data["is_stigmatizing"].notna()]
            results[model_variant][condition] = (
                valid_condition["is_stigmatizing"].mean() if len(valid_condition) > 0 else np.nan
            )

    return results, stigma_df


def extract_tag_categories(tags):
    """Extract the numeric category from each tag code, e.g. '4.Social Distance' -> 4."""
    if not isinstance(tags, (list, np.ndarray)):
        if tags is None or pd.isna(tags):
            return tags  # leave missing values as is
    if hasattr(tags, "__len__") and len(tags) == 0:
        return []
    categories = []
    for tag_dict in tags:
        code_str = tag_dict.get("code", None)
        if code_str is None:
            raise ValueError("Missing 'code' key in tag dict.")
        match = re.match(r"^(\d+)", code_str.strip())
        if not match:
            raise ValueError(f"Tag code does not start with a number: {code_str}")
        categories.append(int(match.group(1)))
    return categories


def compute_tag_category_counts(combined_df: pd.DataFrame):
    """
    For each (model, condition) pair (rows with '_COT' variant), count each tag
    category (1-6).

    Returns:
        model_condition_tag_counts: dict keyed by (model, condition) -> {category: count}
    """
    combined_df = combined_df.copy()
    combined_df["tag_categories"] = combined_df["tags"].apply(extract_tag_categories)

    cot_data = combined_df[combined_df["variant"] == "_COT"]

    category_range = range(1, 7)
    model_condition_tag_counts = {}

    unique_model_conditions = cot_data[["model", "condition"]].drop_duplicates()

    for _, row in unique_model_conditions.iterrows():
        model = row["model"]
        condition = row["condition"]
        selection = cot_data[(cot_data["model"] == model) & (cot_data["condition"] == condition)]
        all_cats = []
        for cats in selection["tag_categories"]:
            if isinstance(cats, list):
                all_cats.extend(cats)
        counts = {cat: all_cats.count(cat) for cat in category_range}
        model_condition_tag_counts[(model, condition)] = counts

    return model_condition_tag_counts


# =============================================================================
# Plots
# =============================================================================

def plot_grouped_bars_by_condition(
    df: pd.DataFrame,
    value_column: str,
    ylabel: str,
    save_path: str,
    annotate: bool = False,
    figsize: tuple = (14, 7),
    dpi: int = 300,
):
    """
    Grouped bar plot of the summed value_column per condition, per model
    (model_variant ending with '_COT').
    """
    sums = df.groupby(["condition", "model_variant"])[value_column].sum().reset_index()
    plot_data = sums[sums["model_variant"].str.endswith("_COT")]

    conditions = plot_data["condition"].unique()
    model_variants = plot_data["model_variant"].unique()
    num_models = len(model_variants)
    x = np.arange(len(conditions))

    formatted_conditions = [format_condition_label(c) for c in conditions]

    plt.figure(figsize=figsize)

    width = 0.8 / num_models

    for idx, model in enumerate(model_variants):
        data_for_model = plot_data[plot_data["model_variant"] == model]
        values_for_model = []
        for cond in conditions:
            val = data_for_model[data_for_model["condition"] == cond][value_column]
            values_for_model.append(val.values[0] if not val.empty else 0)
        color = MODEL_COLOR_CYCLE[idx % len(MODEL_COLOR_CYCLE)]
        bars = plt.bar(x + idx * width, values_for_model, width=width,
                       label=prettify_model_name(model), color=color)

        if annotate:
            for bar, val in zip(bars, values_for_model):
                plt.text(
                    bar.get_x() + bar.get_width() / 2,
                    bar.get_height(),
                    f"{val}",
                    ha="center",
                    va="bottom",
                    fontsize=14,
                )

    plt.ylabel(ylabel, fontsize=16)
    plt.xticks(x + width * (num_models - 1) / 2, formatted_conditions, rotation=0, fontsize=14)
    plt.yticks(fontsize=14)

    plt.legend(
        fontsize=14,
        loc="upper center",
        bbox_to_anchor=(0.5, -0.11),
        ncol=num_models,
        frameon=False,
    )

    plt.tight_layout()
    plt.savefig(save_path, dpi=dpi, bbox_inches="tight")
    plt.close()
    print(f"Saved {save_path}")


def plot_stigma_rate_by_variant(stigma_scores: dict, save_path: str, dpi: int = 300, figsize: tuple = (13, 7)):
    """
    Grouped bar plot of stigma rates across variant groups (Baseline, COT, NTCOT,
    reasoning analysis COT, reasoning analysis NTCOT), one bar per model.
    """
    variant_groups = [
        ("Baseline", lambda name: not name.endswith("_COT") and not name.endswith("_NTCOT"), "overall"),
        ("COT", lambda name: name.endswith("_COT"), "overall"),
        ("NTCOT", lambda name: name.endswith("_NTCOT"), "overall"),
        ("reasoning analysis\ncot", lambda name: name.endswith("_COT"), "reasoning_analysis"),
        ("reasoning analysis\nntcot", lambda name: name.endswith("_NTCOT"), "reasoning_analysis"),
    ]

    def get_model_basename(name):
        if name.endswith("_COT"):
            return name[:-4]
        if name.endswith("_NTCOT"):
            return name[:-6]
        return name

    basenames = []
    for k in stigma_scores.keys():
        bn = get_model_basename(k)
        if bn not in basenames:
            basenames.append(bn)

    palette_default = ["#2166ac", "#b2182b", "#6a3d9a", "#4daf4a", "#ff7f00", "#e7298a", "#a6cee3", "#fdbf6f"]
    color_map = {
        bn: MODEL_COLORS.get(bn, palette_default[i % len(palette_default)])
        for i, bn in enumerate(basenames)
    }

    y = []
    for _, variant_matcher, value_key in variant_groups:
        row = {}
        for bn in basenames:
            possibles = [k for k in stigma_scores if get_model_basename(k) == bn and variant_matcher(k)]
            row[bn] = stigma_scores[possibles[0]].get(value_key, np.nan) if possibles else np.nan
        y.append(row)

    x_labels = [vlabel for vlabel, _, _ in variant_groups]
    n_groups = len(x_labels)
    n_bars = len(basenames)

    heights_per_group = np.zeros((n_groups, n_bars))
    for i, row in enumerate(y):
        for j, bn in enumerate(basenames):
            heights_per_group[i, j] = row[bn]

    fig, ax = plt.subplots(figsize=figsize)
    x = np.arange(n_groups)
    total_bar_cluster_width = 0.89
    bar_width = total_bar_cluster_width / n_bars

    for i, bn in enumerate(basenames):
        pos = x - total_bar_cluster_width / 2 + bar_width / 2 + i * bar_width
        bars = ax.bar(pos, heights_per_group[:, i], bar_width, color=color_map[bn], alpha=0.8)
        for bar in bars:
            height = bar.get_height()
            if not np.isnan(height):
                ax.text(
                    bar.get_x() + bar.get_width() / 2,
                    height + 0.02,
                    f"{height:.2f}",
                    ha="center", va="bottom", fontsize=11,
                    fontweight="bold",
                )

    ax.set_xticks(x)
    ax.set_xticklabels(x_labels, fontweight="bold", fontsize=12)
    ax.set_ylabel("Stigma Rate", fontweight="bold", fontsize=12)
    ax.set_xlabel("Variant", fontweight="bold", fontsize=12)
    ax.set_title("Overall and Reasoning Stigma Rates by Variant and Model", fontweight="bold", fontsize=15)
    ax.set_ylim(0, 1)
    ax.grid(True, alpha=0.25, axis="y")
    for label in ax.get_yticklabels():
        label.set_fontweight("bold")
        label.set_fontsize(12)

    model_legend_patches = [
        mpatches.Patch(facecolor=color_map[bn], edgecolor="black", label=prettify_model_name(bn))
        for bn in basenames
    ]
    legend = plt.legend(handles=model_legend_patches, title="Model", fontsize=11, title_fontsize=12, loc="upper left")
    plt.setp(legend.get_title(), fontweight="bold")
    for text in legend.get_texts():
        text.set_fontweight("bold")

    plt.tight_layout()
    fig.savefig(save_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {save_path}")


def plot_tag_category_stacked_bars(
    model_condition_tag_counts: Dict,
    save_path: str,
    figsize: tuple = (21, 19),
    bar_width: float = 0.7,
    dpi: int = 300,
):
    """
    Stacked bar charts of tag categories by model and condition, in a 3x3 grid
    with the last subplot used for the legend.
    """
    custom_colors = ["#225240", "#96e092", "#5faa41ff", "#397a4a", "#0b56cbff", "#6d9eebff"]
    category_range = range(1, 7)

    models = sorted({k[0] for k in model_condition_tag_counts.keys()})
    raw_conditions = sorted({k[1] for k in model_condition_tag_counts.keys()})

    def capitalize_words(s):
        return " ".join(word.capitalize() for word in s.split())

    condition_cap_map = {cond: capitalize_words(cond) for cond in raw_conditions}

    def nice_model_label(modelname):
        mapping = {
            "claude-opus-45": "Claude \nOpus 4.5",
            "claude-sonnet-4": "Claude \nSonnet 4",
            "llama3.3-70b": "LLaMA\n3.3 70B",
            "deepseek-v31": "DeepSeek\nv3.1",
        }
        return mapping.get(modelname, modelname)

    x_labels = [nice_model_label(m) for m in models]

    # Global max y across subplots
    max_y = 0
    for orig_condition in raw_conditions:
        for m in models:
            key = (m, orig_condition)
            total = sum(model_condition_tag_counts.get(key, {}).get(cat, 0) for cat in category_range)
            max_y = max(max_y, total)

    yticks = list(range(0, 71, 10))
    ytick_max = int(np.ceil(max_y * 1.10 / 10.0)) * 10
    if ytick_max <= 70:
        ytick_max = 80

    nrows, ncols = 3, 3
    num_conditions = len(raw_conditions)
    fig, axes = plt.subplots(nrows, ncols, figsize=figsize)
    axes = axes.flatten()
    plot_axes_count = nrows * ncols - 1  # last axes slot is the legend

    for idx, orig_condition in enumerate(raw_conditions[:plot_axes_count]):
        ax = axes[idx]
        bottoms = np.zeros(len(models))
        bar_centers = np.linspace(0, len(models) - 1, len(models)) if len(models) > 1 else np.array([0.0])

        top_of_bars = bottoms
        for cat_idx, cat in enumerate(category_range):
            cat_counts = [
                model_condition_tag_counts.get((m, orig_condition), {}).get(cat, 0) for m in models
            ]
            ax.bar(
                bar_centers,
                cat_counts,
                bottom=bottoms,
                color=custom_colors[cat_idx % len(custom_colors)],
                width=bar_width,
                align="center",
            )
            bottoms += np.array(cat_counts)
            if cat_idx == len(category_range) - 1:
                top_of_bars = bottoms.copy()

        # Total annotation above the bars
        for xpos, total in zip(bar_centers, top_of_bars):
            if total > 0:
                ax.annotate(
                    f"{int(total)}",
                    (xpos, total + 0.02 * max_y),
                    ha="center",
                    va="bottom",
                    fontsize=14,
                    color="black",
                )

        ax.set_title(condition_cap_map[orig_condition], fontweight="normal", fontsize=18)

        ax.set_yticks(yticks)
        ax.tick_params(axis="y", which="both", length=5, width=1, direction="out", left=True, right=False)

        if idx % ncols == 0:
            ax.set_ylabel("Total Stigma Count", fontsize=18)
            ax.set_yticklabels([str(t) for t in yticks])
            for label in ax.get_yticklabels():
                label.set_fontsize(18)
        else:
            ax.set_yticklabels(["" for _ in yticks])

        ax.set_xticks(bar_centers)
        ax.set_xticklabels(x_labels, rotation=0, ha="center", fontsize=16)
        ax.set_ylim(0, ytick_max)
        ax.set_xlim(bar_centers[0] - bar_width / 2 - 0.1, bar_centers[-1] + bar_width / 2 + 0.1)

        # Hide top and right spines
        for spine_name, spine in ax.spines.items():
            spine.set_visible(spine_name not in ["top", "right"])

    # Turn off unused axes except the legend slot
    for idx in range(num_conditions, plot_axes_count):
        axes[idx].axis("off")

    # Last axis: legend
    legend_ax = axes[-1]
    legend_ax.set_xticks([])
    legend_ax.set_yticks([])
    for side in legend_ax.spines.values():
        side.set_visible(False)

    handles = [
        mpatches.Patch(color=custom_colors[i % len(custom_colors)], label=TAG_CATEGORY_LABELS[i])
        for i in range(len(TAG_CATEGORY_LABELS))
    ]
    legend_ax.legend(
        handles=handles,
        loc="center",
        ncol=1,
        fontsize=20,
        frameon=False,
        labelspacing=1,
    )

    plt.tight_layout(rect=[0, 0.03, 1, 1])
    plt.savefig(save_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {save_path}")


def plot_stigma_severity_bars(combined_df: pd.DataFrame, save_path: str, figsize: tuple = (28, 10), dpi: int = 300):
    """Bar charts of total stigma severity by model, one panel per condition (COT only)."""
    cot_data = combined_df[combined_df["variant"] == "_COT"]

    models = sorted(cot_data["model"].unique())
    conditions = sorted(cot_data["condition"].unique())

    model_condition_severity = {
        (model, condition): cot_data[
            (cot_data["model"] == model) & (cot_data["condition"] == condition)
        ]["stigma_severity"].sum()
        for model in models
        for condition in conditions
    }

    def nice_model_label(modelname):
        mapping = {
            "claude-opus-45": "claude\nopus 4.5",
            "claude-sonnet-4": "claude\nsonnet 4",
        }
        return mapping.get(modelname, modelname)

    x_labels = [nice_model_label(m) for m in models]

    fig, axes = plt.subplots(2, 4, figsize=figsize, sharey=True)
    axes = axes.flatten()

    max_y = max(model_condition_severity.values()) * 1.10 if model_condition_severity else 1

    for idx, condition in enumerate(conditions):
        ax = axes[idx]
        values = [model_condition_severity.get((model, condition), 0) for model in models]
        bars = ax.bar(np.arange(len(models)), values, color="#397a4a")
        for bar, val in zip(bars, values):
            if val > 0:
                ax.annotate(
                    f"{int(val)}",
                    (bar.get_x() + bar.get_width() / 2, val + 0.02 * max_y),
                    ha="center",
                    va="bottom",
                    fontsize=12,
                    color="black",
                    fontweight="bold",
                )
        ax.set_title(condition, fontweight="bold", fontsize=12)
        ax.set_ylabel("Total Stigma Severity", fontweight="bold", fontsize=12)
        ax.set_xticks(np.arange(len(models)))
        ax.set_xticklabels(x_labels, rotation=0, ha="center", fontsize=12, fontweight="bold")
        ax.set_ylim(0, max_y)
        for label in ax.get_yticklabels():
            label.set_fontweight("bold")

    # Remove unused subplots if there are fewer than 8 conditions
    for ax in axes[len(conditions):]:
        ax.axis("off")

    plt.tight_layout(rect=[0, 0, 1, 0.98])
    plt.savefig(save_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {save_path}")


# =============================================================================
# Main
# =============================================================================

def main():
    repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    parser = argparse.ArgumentParser(description="Generate the paper figures from tagged results.")
    parser.add_argument("--results-dir", default=os.path.join(repo_root, "results"), help="Directory with experiment results")
    parser.add_argument("--output-dir", default=os.path.join(repo_root, "figures"), help="Directory to write figures to")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    combined_df = load_all_data(args.results_dir)

    stigma_scores, _ = compute_stigma_scores(combined_df)
    model_condition_tag_counts = compute_tag_category_counts(combined_df)

    plot_stigma_rate_by_variant(
        stigma_scores,
        save_path=os.path.join(args.output_dir, "figure1_stigma_rate_by_variant.png"),
    )
    plot_grouped_bars_by_condition(
        combined_df,
        value_column="how_many_stigma",
        ylabel="Total stigma count",
        annotate=True,
        save_path=os.path.join(args.output_dir, "figure2_stigma_counts_by_condition.png"),
    )
    plot_tag_category_stacked_bars(
        model_condition_tag_counts,
        save_path=os.path.join(args.output_dir, "figure3_tag_categories_by_model_and_condition.png"),
    )
    plot_grouped_bars_by_condition(
        combined_df,
        value_column="stigma_severity",
        ylabel="Total stigma severity",
        annotate=False,
        save_path=os.path.join(args.output_dir, "figure4_stigma_severity_by_condition.png"),
    )
    plot_stigma_severity_bars(
        combined_df,
        save_path=os.path.join(args.output_dir, "figure5_stigma_severity_by_model_and_condition.png"),
    )

    print("Done.")


if __name__ == "__main__":
    main()
