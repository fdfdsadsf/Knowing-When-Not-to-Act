# NEXT CELL:
# Publication-ready TABLE instead of trajectory figures
#
# Requires:
#   results
#   OUTDIR
#
# Expected methods:
#   DP0, DPO, PG1, PG2, PG4, PGstar
#
# All objectives are evaluated under the original eta=0
# environment. LOWER IS BETTER.

import math
import numpy as np
import pandas as pd
from pathlib import Path

try:
    from IPython.display import display
except ImportError:
    display = print


# 0. Method order / labels

METHOD_ORDER = [
    "DP0",
    "DPO",
    "PG1",
    "PG2",
    "PG4",
    "PGstar",
]

METHOD_LABEL = {
    "DP0":    r"DP $\eta=0$",
    "DPO":    r"$u_{\theta}$",
    "PG1":    r"$\mathcal{P}_{\eta}^{[1]}$",
    "PG2":    r"$\mathcal{P}_{\eta}^{[2]}$",
    "PG4":    r"$\mathcal{P}_{\eta}^{[4]}$",
    "PGstar": r"$\mathcal{P}_{\eta}^{*}$",
}


# Robustly detect DP key in case previous cell used
# "DP", "DP_eta0", etc.

DP_CANDIDATES = [
    "DP0",
    "DP",
    "DP_eta0",
    "DP eta=0",
    "DP_eta=0",
]

dp_key = None

for k in DP_CANDIDATES:
    if k in results:
        dp_key = k
        break

if dp_key is None:
    raise KeyError(
        "Could not find the eta=0 DP result in `results`. "
        f"Available keys: {list(results.keys())}"
    )

# normalize method list against actually available results
resolved_methods = []

for key in METHOD_ORDER:

    if key == "DP0":
        resolved_methods.append(dp_key)

    elif key in results:
        resolved_methods.append(key)

missing = [
    k for k in ["DPO", "PG1", "PG2", "PG4", "PGstar"]
    if k not in results
]

if missing:
    raise KeyError(
        f"Missing expected methods: {missing}\n"
        f"Available keys: {list(results.keys())}"
    )


# 1. Helpers

def mean_se(x):
    x = np.asarray(x, dtype=float)

    mean = float(np.mean(x))

    if len(x) > 1:
        se = float(
            np.std(x, ddof=1)
            / math.sqrt(len(x))
        )
    else:
        se = np.nan

    return mean, se


def latex_pm(mean, se, digits=4):
    if np.isnan(se):
        return f"${mean:.{digits}f}$"

    return (
        f"${mean:.{digits}f}"
        rf"\pm{se:.{digits}f}$"
    )


def plain_pm(mean, se, digits=6):
    if np.isnan(se):
        return f"{mean:.{digits}f}"

    return (
        f"{mean:.{digits}f} "
        f"± {se:.{digits}f}"
    )


# 2. DP reference

dp_obj = np.asarray(
    results[dp_key]["objective"],
    dtype=float,
)

n_paths = len(dp_obj)

dp_mean = float(
    np.mean(dp_obj)
)

if n_paths < 2:
    raise RuntimeError(
        "At least two held-out paths are needed "
        "for standard errors."
    )


# 3. Build detailed numerical table

rows = []

for method in resolved_methods:

    r = results[method]

    obj = np.asarray(
        r["objective"],
        dtype=float,
    )

    if len(obj) != n_paths:
        raise RuntimeError(
            f"{method}: expected {n_paths} common paths, "
            f"found {len(obj)}."
        )

    release = np.asarray(
        r["release_cost"],
        dtype=float,
    )

    running = np.asarray(
        r["running_cost"],
        dtype=float,
    )

    final_dist = np.asarray(
        r["final_distance"],
        dtype=float,
    )


    # Terminal penalty decomposition
    #
    # J0 =
    #   running state cost
    # + singular release cost
    # + terminal penalty

    terminal = (
        obj
        - running
        - release
    )


    # Because every method is evaluated on the SAME
    # held-out paths, use a PAIRED difference to DP.

    gap = (
        obj
        - dp_obj
    )


    # Pathwise relative gap.
    #
    # We report the ratio of mean costs separately below;
    # paired absolute gap SE remains statistically cleaner.

    obj_mean, obj_se = mean_se(obj)
    gap_mean, gap_se = mean_se(gap)

    run_mean, run_se = mean_se(running)
    rel_mean, rel_se = mean_se(release)
    terminal_mean, terminal_se = mean_se(terminal)
    dist_mean, dist_se = mean_se(final_dist)


    relative_gap_pct = (
        100.0
        *
        (obj_mean - dp_mean)
        /
        abs(dp_mean)
    )


    display_key = (
        "DP0"
        if method == dp_key
        else method
    )


    rows.append(
        {
            "Method": METHOD_LABEL[display_key],

            "Objective_mean":
                obj_mean,

            "Objective_SE":
                obj_se,

            "Gap_to_DP_mean":
                gap_mean,

            "Gap_to_DP_SE":
                gap_se,

            "Gap_to_DP_pct":
                relative_gap_pct,

            "Running_cost_mean":
                run_mean,

            "Running_cost_SE":
                run_se,

            "Release_cost_mean":
                rel_mean,

            "Release_cost_SE":
                rel_se,

            "Terminal_penalty_mean":
                terminal_mean,

            "Terminal_penalty_SE":
                terminal_se,

            "Final_distance_mean":
                dist_mean,

            "Final_distance_SE":
                dist_se,

            "Eval_time_sec":
                float(
                    r.get(
                        "eval_time_sec",
                        np.nan,
                    )
                ),
        }
    )


detailed_df = pd.DataFrame(rows)


# 4. Human-readable notebook table

pretty_rows = []

for _, row in detailed_df.iterrows():

    pretty_rows.append(
        {
            "Method":
                row["Method"],

            r"$J_0$ ↓":
                plain_pm(
                    row["Objective_mean"],
                    row["Objective_SE"],
                    digits=5,
                ),

            "Gap to DP ↓":
                plain_pm(
                    row["Gap_to_DP_mean"],
                    row["Gap_to_DP_SE"],
                    digits=5,
                ),

            "Rel. gap (%) ↓":
                f'{row["Gap_to_DP_pct"]:.2f}',

            "Running cost ↓":
                plain_pm(
                    row["Running_cost_mean"],
                    row["Running_cost_SE"],
                    digits=5,
                ),

            "Release cost ↓":
                plain_pm(
                    row["Release_cost_mean"],
                    row["Release_cost_SE"],
                    digits=5,
                ),

            "Terminal penalty ↓":
                plain_pm(
                    row["Terminal_penalty_mean"],
                    row["Terminal_penalty_SE"],
                    digits=5,
                ),

            "Final distance ↓":
                plain_pm(
                    row["Final_distance_mean"],
                    row["Final_distance_SE"],
                    digits=5,
                ),
        }
    )


pretty_df = pd.DataFrame(
    pretty_rows
)


print("")
print("=" * 120)
print(
    "TWO-RESERVOIR ORIGINAL eta=0 PERFORMANCE "
    "(lower is better)"
)
print("=" * 120)
print(
    f"Common held-out paths: {n_paths}"
)
print(
    "Gap to DP is computed PATHWISE using "
    "the common random numbers."
)
print("")

display(pretty_df)


# 5. Compact MAIN-PAPER table
#
# Recommended main-paper version:
#   objective
#   paired DP gap
#   release cost
#   final distance
#
# Running / terminal decomposition can go to appendix.

main_rows = []

for _, row in detailed_df.iterrows():

    main_rows.append(
        {
            "Method":
                row["Method"],

            r"$J_0$ ↓":
                latex_pm(
                    row["Objective_mean"],
                    row["Objective_SE"],
                    digits=4,
                ),

            r"$J_0-J_0^{\rm DP}$ ↓":
                latex_pm(
                    row["Gap_to_DP_mean"],
                    row["Gap_to_DP_SE"],
                    digits=4,
                ),

            r"Relative gap (\%) ↓":
                f'{row["Gap_to_DP_pct"]:.2f}',

            r"Release cost ↓":
                latex_pm(
                    row["Release_cost_mean"],
                    row["Release_cost_SE"],
                    digits=4,
                ),

            r"Final distance ↓":
                latex_pm(
                    row["Final_distance_mean"],
                    row["Final_distance_SE"],
                    digits=4,
                ),
        }
    )


main_df = pd.DataFrame(
    main_rows
)


print("")
print("=" * 120)
print("COMPACT MAIN-PAPER TABLE")
print("=" * 120)

display(main_df)


# 6. Save CSVs

try:
    OUTDIR
except NameError:
    OUTDIR = Path(
        "reservoir_release_refinement"
    )

OUTDIR = Path(OUTDIR)

OUTDIR.mkdir(
    parents=True,
    exist_ok=True,
)


detailed_csv = (
    OUTDIR
    /
    "reservoir_dp_comparison_detailed.csv"
)

pretty_csv = (
    OUTDIR
    /
    "reservoir_dp_comparison_pretty.csv"
)


detailed_df.to_csv(
    detailed_csv,
    index=False,
)

pretty_df.to_csv(
    pretty_csv,
    index=False,
)


# 7. Generate publication-ready LaTeX

latex_lines = [
    r"\begin{table}[t]",
    r"\centering",
    r"\small",
    r"\setlength{\tabcolsep}{4.5pt}",
    r"\caption{"
    r"Two-reservoir singular-control benchmark. "
    r"All learned controllers are evaluated under the original "
    r"$\eta=0$ dynamics on common held-out disturbance paths. "
    r"DP solves the same discretized original problem and is used "
    r"as the low-dimensional numerical ground truth. "
    r"Reported uncertainties are standard errors across held-out paths. "
    r"The DP gap is paired pathwise; lower is better."
    r"}",
    r"\label{tab:reservoir-dp-main}",
    r"\begin{tabular}{lccccc}",
    r"\toprule",
    (
        r"Method"
        r" & $J_0$"
        r" & $J_0-J_0^{\rm DP}$"
        r" & Gap (\%)"
        r" & Release cost"
        r" & Final distance \\"
    ),
    r"\midrule",
]


for _, row in detailed_df.iterrows():

    method = row["Method"]

    obj_txt = latex_pm(
        row["Objective_mean"],
        row["Objective_SE"],
        digits=4,
    )

    gap_txt = latex_pm(
        row["Gap_to_DP_mean"],
        row["Gap_to_DP_SE"],
        digits=4,
    )

    rel_txt = (
        f'{row["Gap_to_DP_pct"]:.2f}'
    )

    release_txt = latex_pm(
        row["Release_cost_mean"],
        row["Release_cost_SE"],
        digits=4,
    )

    dist_txt = latex_pm(
        row["Final_distance_mean"],
        row["Final_distance_SE"],
        digits=4,
    )


    latex_lines.append(
        f"{method}"
        f" & {obj_txt}"
        f" & {gap_txt}"
        f" & {rel_txt}"
        f" & {release_txt}"
        f" & {dist_txt}"
        r" \\"
    )


latex_lines += [
    r"\bottomrule",
    r"\end{tabular}",
    r"\end{table}",
]


latex_table = "\n".join(
    latex_lines
)


latex_path = (
    OUTDIR
    /
    "reservoir_dp_comparison_table.tex"
)


latex_path.write_text(
    latex_table,
    encoding="utf-8",
)


print("")
print("=" * 120)
print("LATEX TABLE")
print("=" * 120)
print(latex_table)


print("")
print("[saved]")
print(" ", detailed_csv)
print(" ", pretty_csv)
print(" ", latex_path)
