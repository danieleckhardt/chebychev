import configparser
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import matplotlib.pyplot as plt
# from matplot2tikz import save as tikz_save


# =============================================================================
# USER SETTINGS
# =============================================================================

example = 201 # Enter the sample numbers: 201, 301
methods = ["poly_krylov", "rat_krylov", "cheb"]
betas = [0.001,0.1,0.01] # Betalist, see the corresponding configuration files in the “config” folder
# 201: [0.01,0.1,0.001]
# 301: [1,0.1,0.01]


# ============================================================
# Configuration Utilities
# ============================================================

def load_config(example: int) -> configparser.ConfigParser:
    cfg = configparser.ConfigParser()

    config_map = {
        101: "config/example101.ini",
        201: "config/example201.ini",
        301: "config/example301.ini",
    }

    if example not in config_map:
        raise ValueError(f"Unknown example {example}")

    cfg.read(config_map[example])
    return cfg


def geti(cfg, section, key, default=0):
    try:
        return cfg.getint(section, key)
    except Exception:
        return int(default)


def getf(cfg, section, key, default=0.0):
    try:
        return cfg.getfloat(section, key)
    except Exception:
        return float(default)


def gets(cfg, section, key, default=""):
    return cfg.get(section, key, fallback=default)


# ============================================================
# File Handling
# ============================================================

def format_float(value: float) -> str:
    """Remove trailing zeros (1.0 → 1, 0.100 → 0.1)."""
    return f"{value:g}"


def build_error_filename(
    cfg: configparser.ConfigParser,
    scheme: str,
    beta: float
) -> str:
    if not scheme:
        raise ValueError("Scheme must be provided.")

    h = geti(cfg, 'Space_Discretization', 'initial_refinement')
    deg = geti(cfg, 'Space_Discretization', 'fe_degree')
    tau = getf(cfg, 'Data', 'tau')

    tau_str = format_float(tau)
    beta_str = format_float(beta)

    name = f"h{h}_deg{deg}_tau{tau_str}_beta{beta_str}"
    return f"{scheme}/{name}"


def read_xy(file_path: Path) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    if not file_path.exists():
        raise FileNotFoundError(f"Missing file: {file_path}")

    data = np.loadtxt(file_path, skiprows=1)
    return data[:, 0], data[:, 1], data[:, 2]


# ============================================================
# Plotting Utilities
# ============================================================

METHOD_LABELS = {
    "poly_krylov": "Polynomial Krylov",
    "rat_krylov": "Rational Krylov",
    "cheb": "Chebyshev",
}


def setup_axis(ax, xlabel: str):
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xlabel(xlabel)
    ax.set_ylabel("error")
    ax.grid(True, which='both', linestyle=':')


def create_marker_map(methods: List[str]) -> Dict[str, str]:
    marker_list = ['s', '^', 'o', 'D', 'v', '*', 'x', '+']
    return {
        method: marker_list[i % len(marker_list)]
        for i, method in enumerate(methods)
    }


def add_series(ax, x, y, method: str, markers: Dict[str, str]):
    if np.any(x <= 0) or np.any(y <= 0):
        raise ValueError("Log-scale plotting requires strictly positive values.")

    label = METHOD_LABELS.get(method, method)

    ax.plot(
        x,
        y,
        marker=markers[method],
        linestyle='-',
        linewidth=1.2,
        markersize=4.0,
        label=label
    )


# ============================================================
# LaTeX Table Export
# ============================================================

def export_latex_table(
    output_path: Path,
    betas: List[float],
    methods: List[str],
    cfg: configparser.ConfigParser,
    base_folder: Path
):
    """
    Single LaTeX table containing ALL betas and ALL iterations.
    """

    # Read everything first
    data = {}
    max_len_per_beta = {}

    for beta in betas:
        beta_data = {}
        max_len = 0

        for method in methods:
            fn = build_error_filename(cfg, method, beta)
            err, mv, time = read_xy(base_folder / fn)
            beta_data[method] = (err, mv, time)
            max_len = max(max_len, len(err))

        data[beta] = beta_data
        max_len_per_beta[beta] = max_len

    with open(output_path, "w") as f:

        f.write("\\begin{table}[ht]\n")
        f.write("\\centering\n")

        col_format = "c" + "|ccc" * len(methods)
        f.write(f"\\begin{{tabular}}{{{col_format}}}\n")
        f.write("\\hline\n")

        # Header row 1
        header1 = "$\\delta$ "
        for m in methods:
            label = METHOD_LABELS.get(m, m)
            header1 += f"& \\multicolumn{{3}}{{c}}{{{label}}} "
        header1 += "\\\\"
        f.write(header1)

        # Header row 2
        header2 = " "
        for _ in methods:
            header2 += "& err & mv & time (s) "
        header2 += "\\\\"

        f.write(header2)
        f.write("\\hline")

        # Write all rows
        for beta in betas:

            beta_str = format_float(beta)
            max_len = max_len_per_beta[beta]

            for i in range(1, max_len, 2):
                if i == 1:
                    row = f"{beta_str}  "
                else:
                    row = " "


                for method in methods:

                    err, mv, time = data[beta][method]

                    if i < len(err):
                        row += (
                            f"& {err[i]:.2e} "
                            f"& {int(mv[i])} "
                            f"& {time[i]:.2e} "
                        )
                    else:
                        row += "& - & - & - "

                row += "\\\\"
                f.write(row)
            f.write("\\hline\n")

        f.write("\\end{tabular}\n")
        f.write(
            "\\caption{All iterations for all $\\beta$ values. "
            "Error, matrix-vector multiplications (mv), and runtime (s).}\n"
        )
        f.write("\\end{table}\n")


# ============================================================
# Main Routine
# ============================================================

def main():

    cfg = load_config(example)

    ex = gets(cfg, 'Calculation_Mode', 'example', str(example))
    base_folder = Path(f"output/errors/Example{ex}")
    output_dir = base_folder / "tables_figures"
    output_dir.mkdir(parents=True, exist_ok=True)

    markers = create_marker_map(methods)

    for beta in betas:

        fig_time, ax_time = plt.subplots()
        fig_mv, ax_mv = plt.subplots()

        setup_axis(ax_time, r"time in $s$")
        setup_axis(ax_mv, "Matrix-Vector multiplications")

        for method in methods:
            fn = build_error_filename(cfg, method, beta)
            err, mv, time = read_xy(base_folder / fn)

            add_series(ax_time, time, err, method, markers)
            add_series(ax_mv, mv, err, method, markers)

        ax_time.legend(loc='lower right', fontsize='small')
        ax_mv.legend(loc='lower right', fontsize='small')

        beta_str = format_float(beta)

        # tikz_save(output_dir / f"time_error{beta_str}.tikz", figure=fig_time)
        # tikz_save(output_dir / f"mv_error{beta_str}.tikz", figure=fig_mv)

        fig_time.savefig(output_dir / f"time_error{beta_str}.png", dpi=300)
        fig_mv.savefig(output_dir / f"mv_error{beta_str}.png", dpi=300)

        plt.close(fig_time)
        plt.close(fig_mv)

    export_latex_table(
        output_dir / "summary_table_triplets.tex",
        betas,
        methods,
        cfg,
        base_folder
    )

    print(f"Results written to: {output_dir}")


if __name__ == "__main__":
    main()