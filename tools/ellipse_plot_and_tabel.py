import configparser
from pathlib import Path
from typing import List, Tuple
import numpy as np
import matplotlib.pyplot as plt
from matplot2tikz import save as tikz_save
from dataclasses import dataclass
from typing import List

# =============================================================================
# USER SETTINGS
# =============================================================================

EXAMPLE: int = 301 # Enter the sample numbers: 101, 201, 301
BETAS: List[float] = [1,0.1,0.01]   # Betalist, see the corresponding configuration files in the “config” folder
# 101: [0.4,0.03,0.01]
# 201: [0.01,0.1,0.001]
# 301: [1,0.1,0.01]

OUTPUT_FILENAME: str = "ellipse_steps.tex"

# =============================================================================
# DATA MODEL
# =============================================================================

@dataclass(frozen=True)
class EllipseRecord:
    beta: float
    radius_x: float
    radius_y: float
    center: float
    time_s: float

# =============================================================================
# CONFIG HANDLING
# =============================================================================

class Config:
    def __init__(self, example: int) -> None:
        self._cfg = configparser.ConfigParser()
        self._load(example)

    def _load(self, example: int) -> None:
        config_map = {
            101: "config/example101.ini",
            201: "config/example201.ini",
            301: "config/example301.ini",
        }

        if example not in config_map:
            raise ValueError(f"Unknown example {example}")

        config_path = Path(config_map[example])

        if not config_path.exists():
            raise FileNotFoundError(f"Missing config file: {config_path}")

        self._cfg.read(config_path)

    def get(self, section: str, key: str, cast=str, default=None):
        try:
            return cast(self._cfg.get(section, key))
        except (configparser.NoOptionError,
                configparser.NoSectionError,
                ValueError):
            return default
# =============================================================================
# FILE NAME BUILDER
# =============================================================================

def build_error_filename(cfg: Config, beta: float) -> Path:
    h   = cfg.get("Space_Discretization", "initial_refinement", int, 0)
    deg = cfg.get("Space_Discretization", "fe_degree", int, 0)
    tau = cfg.get("Data", "tau", float, 0.0)

    name = f"h{h}_deg{deg}_tau{tau}_beta{beta}_ell_info"
    return Path("cheb") / name

# =============================================================================
# PARSING
# =============================================================================

def parse_dataset(line1: str, line2: str, beta: float) -> EllipseRecord:
    parts = line1.strip().split(";")

    if len(parts) < 2:
        raise ValueError("Malformed dataset line.")

    radius_x = float(parts[0])
    radius_y = float(parts[1])
    center   = float(parts[-1])
    time_s   = float(line2.strip())

    return EllipseRecord(beta, radius_x, radius_y, center, time_s)


def read_file_without_ref(path: Path, beta: float) -> List[EllipseRecord]:
    if not path.exists():
        raise FileNotFoundError(f"Missing error file: {path}")

    with open(path, "r") as f:
        lines = f.readlines()

    if len(lines) % 2 != 0:
        raise ValueError(f"File {path} has an odd number of lines.")

    records: List[EllipseRecord] = []

    for i in range(0, len(lines), 2):
        records.append(parse_dataset(lines[i], lines[i + 1], beta))

    return records

####
#### when ref ellipse is given
####

def parse_ellipse(raw: str) -> List[float]:
    return [float(x) for x in raw.strip("[]").split(",")]


def parse_points(raw: str, ref  = False ) -> List[List[float]]:
    raw = raw.strip("[]")
    if not raw:
        return []

    points = []
    for p in raw.split("),("):
        p = p.strip("()")
        points.append([float(x) for x in p.split(",")])
    return points


def parse_row(row: str) -> Tuple[int, float, List[float], List[List[float]]]:
    """
    Expected format:
    deg;rad;[ellipse...];[(x,y,flag),(x,y,flag),...]
    """
    deg_str, rad_str, ell_str, pts_str = row.split(";", 3)

    return (
        int(deg_str),
        float(rad_str),
        parse_ellipse(ell_str),
        parse_points(pts_str),
    )


def read_with_ref(path: Path):
    if not path.exists():
        raise FileNotFoundError(path)

    lines = path.read_text().splitlines()
    if len(lines) < 3:
        raise ValueError(f"{path} does not contain enough data")

    first_ellipse = parse_ellipse(lines[0]) 
    parsed = [parse_row(line) for line in lines[1:-3]]

    deg_last, rad_last = lines[-3].split(";", 1)
    deg_last, rad_last = int(deg_last), float(rad_last)

    deg = np.array([p[0] for p in parsed] + [deg_last])
    radius = np.array([p[1] for p in parsed] + [rad_last])
    ellipse = np.array([p[2] for p in parsed], dtype=object)
    points = [p[3] for p in parsed]

    ref_point_string = lines[-2].split(";",-1)

    ref_points = []
    for p in ref_point_string:
        p = p.strip("()")
        if p != '':
            ref_points.append([float(x) for x in p.split(",")])
    ref_ellipse = parse_ellipse(lines[-1])
    return deg, radius, ellipse, points, ref_points, ref_ellipse, first_ellipse

# =============================================================================
# LATEX TABLE GENERATION
# =============================================================================

def generate_latex_table(data: List[EllipseRecord]) -> str:
    lines = [
        "\\begin{table}[ht]\n",
        "\\centering\n",
        "\\begin{tabular}{c|ccc|c}",
        "\\hline",
        "$\\beta$ & $\\alpha$ & $\\beta$ & $\\gamma$ & Time (s) \\\\",
        "\\hline",
    ]

    for r in data:
        row = (
            f"{r.beta:.2f} & "
            f"{r.radius_x:.1f} & "
            f"{r.radius_y:.1f} & "
            f"{r.center:.2f} & "
            f"{r.time_s:.3f} \\\\"
        )
        lines.append(row)
    lines.append("\\hline")
    lines.append("\\end{tabular}")

    return "\n".join(lines)

#
# with ref
#

def count_new_points(points: List[List[float]]) -> int:
    return sum(p[-1] == 1 for p in points)


def ellipse_changed(ellipse: np.ndarray, i: int) -> bool:
    if i == 0:
        return ellipse[0][1] != 0
    return not np.array_equal(ellipse[i], ellipse[i - 1])


def build_latex_table(
    deg: np.ndarray,
    radius: np.ndarray,
    ellipse: np.ndarray,
    points: List[List[List[float]]],
) -> str:
    headers = [
        r"\texttt{Degree}",
        r"\texttt{Radius}",
        r"\texttt{Points found}",
        r"\texttt{Ellipse changed}",
    ]

    lines = [
        r"\begin{table}[h!]",
        r"\centering",
        r"\begin{tabular}{|c||r|r|r|r|}",
        "Steps & " + " & ".join(headers) + r" \\",
        r"\hline",
    ]

    for i in range(len(deg)):
        is_last = i == len(deg) - 1

        row = [
            str(i),
            str(deg[i]),
            str(radius[i]),
            "-" if is_last else str(count_new_points(points[i])),
            "-" if is_last else ("Yes" if ellipse_changed(ellipse, i) else "No"),
        ]

        lines.append(" & ".join(row) + r" \\")

    lines.extend([
        r"\hline",
        r"\end{tabular}",
        r"\caption{Steps for finding ellipse}",
        rf"\label{{tab:findellipse{EXAMPLE}}}",
        r"\end{table}",
    ])

    return "\n".join(lines)

# =============================================================================
# PLOTTING
# =============================================================================

def plot_complex_plane_steps(
    points: List[List[List[float]]],
    ref_points: List[List[float]],
    ellipse: np.ndarray,
    ref_ellipse: np.ndarray,
    first_ellipse: np.ndarray,
    beta: float,
    steps=None,
    save_dir: Path | None = None,
    show_new_points_only: bool = False,
    figsize=(7, 7),   # unused, kept for compatibility
):

    if steps is None:
        steps = range(len(points) + 1)

    if save_dir is None:
        return

    save_dir.mkdir(parents=True, exist_ok=True)
    tex_path = save_dir / f"ellipses_{beta:.3g}.tikz"

    # -------------------------------------------------
    # Viridis colors (like matplotlib)
    # -------------------------------------------------
    cmap = plt.cm.viridis
    cols = cmap(np.linspace(0, 0.75, len(steps)))

    rgb_colors = []
    for c in cols:
        r = int(round(c[0] * 255))
        g = int(round(c[1] * 255))
        b = int(round(c[2] * 255))
        rgb_colors.append((r, g, b))

    # -------------------------------------------------
    # Collect all points for bounding box
    # -------------------------------------------------
    all_x, all_y = [], []

    def collect_xy(data):
        for x, y in data:
            all_x.append(x)
            all_y.append(y)

    if ref_points:
        collect_xy(ref_points)

    for step_pts in points:
        filtered = step_pts
        if show_new_points_only:
            filtered = [p for p in step_pts if p[-1] == 1]
        collect_xy([p[:2] for p in filtered])

    for e in ellipse:
        a, b, c = e
        all_x.extend([c - a, c + a])
        all_y.extend([-b, b])

    if ref_ellipse is not None:
        a, b, c = ref_ellipse
        all_x.extend([c - a, c + a])
        all_y.extend([-b, b])

    xmin, xmax = min(all_x), max(all_x)
    ymin, ymax = min(all_y), max(all_y)

    pad_x = 0.05 * (xmax - xmin)
    pad_y = 0.05 * (ymax - ymin)

    xmin -= pad_x
    xmax += pad_x
    ymin -= pad_y
    ymax += pad_y

    # -------------------------------------------------
    # Write PGFPlots file
    # -------------------------------------------------
    with open(tex_path, "w") as f:

        # Define colors
        for i, (r, g, b) in enumerate(rgb_colors):
            f.write(f"\\definecolor{{step{i}}}{{RGB}}{{{r},{g},{b}}}\n")
        f.write("\n")

        # Axis
        f.write("\\begin{axis}[\n")
        f.write("axis lines=middle,\n")
        f.write(f"xmin={xmin}, xmax={xmax},\n")
        f.write(f"ymin={ymin}, ymax={ymax},\n")
        f.write("xlabel={$\\Re(z)$},\n")
        f.write("ylabel={$\\Im(z)$},\n")
        f.write("legend style={font=\\small},\n")
        f.write("]\n\n")

        # -------------------------------------------------
        # Loop over steps
        # -------------------------------------------------
        for idx, i in enumerate(steps):

            linewidth = max(0.2, 1.0 - 0.1 * i)

            # --- Ellipse ---
            if i == 0:
                # Reference ellipse
                a, b, c = ref_ellipse
                f.write(rf"""
                % Reference ellipse
                \addplot [
                    domain=0:360,
                    samples=200,
                    dashed,
                    red,
                    line width={linewidth*1.}pt
                ] ({{{a}*cos(x)+{c}}}, {{{b}*sin(x)}});

                \addlegendentry{{$\mathcal{{E}}_{{\text{{best}}}}$}}
                """)

                # Step 0 ellipse
                a, b, c = first_ellipse
                f.write(f"""
                    % Step 0
                    \\addplot [
                        domain=0:360,
                        samples=200,
                        smooth,
                        draw=step{idx},
                        line width={linewidth}pt
                    ] ({{{a}*cos(x)+{c}}}, {{{b}*sin(x)}});
                    \\addlegendentry{{Step 0}}
                    """)
            else:
                a, b, c = ellipse[i - 1]
                f.write(f"""
                    % Step {i}
                    \\addplot [
                        domain=0:360,
                        samples=200,
                        smooth,
                        draw=step{idx},
                        line width={linewidth}pt
                    ] ({{{a}*cos(x)+{c}}}, {{{b}*sin(x)}});
                    \\addlegendentry{{Step {i}}}
                    """)

            # --- Points ---
        for idx, i in enumerate(steps):
            if i == 0:
                if ref_points:
                    f.write("\\addplot [only marks, mark=*, mark options={scale=0.75},  red]\n")
                    f.write("table{%\nx y\n")
                    for x, y in ref_points:
                        f.write(f"{x} {y}\n")
                    f.write("};\n\n")
                    f.write(f"""\\addlegendentry{{EV}}""")
            else:
                step_pts = points[i - 1]

                if show_new_points_only:
                    step_pts = [p for p in step_pts if p[-1] == 1]

                new_pts = [p[:2] for p in step_pts if p[-1] == 1]

                if new_pts:
                    f.write(
                        f"\\addplot [only marks, "
                        f"mark=x, "
                        f"mark options={{scale=1.1}}, "
                        f"draw=step{idx}, "
                        f"forget plot]\n"
                    )
                    f.write("table{%\nx y\n")
                    for x, y in new_pts:
                        f.write(f"{x} {y}\n")
                    f.write("};\n\n")

        f.write("\\end{axis}\n")

    print(f"PGFPlots figure written to {tex_path}")

def plot_complex_plane_png(
    points: List[List[List[float]]],
    ref_points: List[List[float]],
    ellipse: np.ndarray,
    ref_ellipse: np.ndarray,
    first_ellipse: np.ndarray,
    beta: float,
    steps=None,
    save_dir: Path | None = None,
    show_new_points_only: bool = True,
    figsize=(7, 7),
):
    if steps is None:
        steps = range(len(points) + 1)

    fig, ax = plt.subplots(figsize=figsize)

    # Axes lines
    ax.axhline(0, color="black", linewidth=0.8)
    ax.axvline(0, color="black", linewidth=0.8)

    ax.set_xlabel(r"$\Re(z)$")
    ax.set_ylabel(r"$\Im(z)$")

    theta = np.linspace(0, 2 * np.pi, 400)
    colors = plt.cm.viridis(np.linspace(0, 0.85, len(steps)))

    for idx, i in enumerate(steps):
        color = colors[idx]
        linewidth = max(0.6, 2.0 - 0.3 * i)

        # -------------------------------------------------
        # Ellipses
        # -------------------------------------------------
        if i == 0:
            # Reference ellipse
            a, b, c = ref_ellipse
            x = a * np.cos(theta) + c
            y = b * np.sin(theta)
            ax.plot(x, y, "--", color="red",
                    linewidth=linewidth * 1.1,
                    label="Ref")

            # Step 0
            a, b, c = first_ellipse
            x = a * np.cos(theta) + c
            y = b * np.sin(theta)
            ax.plot(x, y,
                    color=color,
                    linewidth=linewidth,
                    label="Step 0")
        else:
            a, b, c = ellipse[i - 1]
            x = a * np.cos(theta) + c
            y = b * np.sin(theta)
            ax.plot(x, y,
                    color=color,
                    linewidth=linewidth,
                    label=f"Step {i}")

    # -------------------------------------------------
    # Points
    # -------------------------------------------------
    for idx, i in enumerate(steps):
        color = colors[idx]
        if i == 0:
            if ref_points:
                pts = np.array(ref_points)
                ax.scatter(pts[:, 0], pts[:, 1],
                           color="red", s=15)
        else:
            step_pts = points[i - 1]
            if show_new_points_only:
                step_pts = [p for p in step_pts if p[-1] == 1]

            new_pts = [p[:2] for p in step_pts if p[-1] == 1]

            if new_pts:
                pts = np.array(new_pts)
                ax.scatter(pts[:, 0], pts[:, 1],
                           marker="x",
                           color=color,
                           s=50)

    ax.grid(True, linestyle=":", linewidth=0.6)
    ax.legend(fontsize=8)
    ax.set_aspect("equal", adjustable="box")
    plt.tight_layout()

    # -------------------------------------------------
    # Save PNG
    # -------------------------------------------------
    if save_dir is not None:
        save_dir.mkdir(parents=True, exist_ok=True)
        png_path = save_dir / f"ellipses_{beta:.3g}.png"
        fig.savefig(png_path, dpi=300, bbox_inches="tight")
        print(f"PNG figure saved to {png_path}")

    plt.close(fig)
# =============================================================================
# FILE WRITING
# =============================================================================

def write_latex_file(content: str, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w") as f:
        f.write(content)
        f.write("\n")
        f.write(
            "\\caption{All iterations for all $\\beta$ values. "
            "Error, matrix-vector multiplications (mv), and runtime (s).}\n"
        )
        f.write("\\end{table}\n")

# =============================================================================
# MAIN
# =============================================================================
def main():
    config = Config(EXAMPLE)

    example_id = config.get("Calculation_Mode", "example", str, "10")
    base_folder = Path(f"output/errors/Example{example_id}")
    output_dir = Path(f"output/errors/Example{EXAMPLE}/tables_figures")
    output_path = output_dir / OUTPUT_FILENAME

    INFO = config.get("Output", "info_ellipse", str)
    if INFO == "true":
        info = True
    else:
        info = False
        
    all_records: List[EllipseRecord] = []

    for beta in BETAS:
        error_file = base_folder / build_error_filename(config, beta)
        if info:
            deg, radius, ellipse, points, ref_points, ref_ellipse, first_ellipse = read_with_ref(error_file)
            plot_complex_plane_steps(points, ref_points, ellipse, ref_ellipse, first_ellipse, beta, save_dir=output_dir)
            plot_complex_plane_png(points, ref_points, ellipse, ref_ellipse, first_ellipse, beta, save_dir=output_dir)
        else:
            records = read_file_without_ref(error_file, beta)
            all_records.extend(records)

    if not info:
        all_records.sort(key=lambda r: r.beta)
        latex_table = generate_latex_table(all_records)
        write_latex_file(latex_table, output_path)
        print(f"LaTeX table written to {output_path}")
if __name__ == "__main__":
    main()