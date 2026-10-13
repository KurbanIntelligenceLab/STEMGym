"""Generate manuscript figures and tables directly from released records and logs."""

from pathlib import Path
import argparse, json
import numpy as np
import h5py
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results/generated"
HERE = Path(__file__).resolve().parent
REPRO_OUT = HERE
plt.rcParams.update(
    {
        "font.family": "DejaVu Sans",
        "font.size": 10,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "pdf.fonttype": 42,
    }
)
WORLDS = [
    "srtio3_clustered_easy",
    "batio3_interface_easy",
    "sige_gradient_clustered_easy",
    "gan_easy",
]
NAMES = ["SrTiO$_3$", "BaTiO$_3$", "SiGe", "GaN"]
POLICIES = [
    "raster_heuristic",
    "raster_trained",
    "random_trained",
    "gp_intensity_trained",
    "gp_defect_trained",
]
LABELS = [
    "Raster + heuristic",
    "Raster + trained",
    "Random + trained",
    "GP intensity + trained",
    "GP defects + trained",
]
COLORS = ["#7c8792", "#126aab", "#df9635", "#31846c", "#9b5e99"]


def write_generated_table(section, name, table):
    """Update the table in its section, retaining surrounding manuscript text."""
    target = OUT / section
    begin = "% BEGIN GENERATED TABLE: " + name
    end = "% END GENERATED TABLE: " + name
    source = target.read_text() if target.exists() else begin + "\n" + end + "\n"
    if not source.count(begin) == source.count(end) == 1:
        raise ValueError(
            "Validation failed: source.count(begin) == source.count(end) == 1"
        )
    start = source.index(begin) + len(begin)
    stop = source.index(end, start)
    updated = source[:start] + "\n" + table.rstrip() + "\n" + source[stop:]
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(updated)


def data_figures():
    worlds = WORLDS + ["pt_nanoparticles_easy"]
    fig = plt.figure(figsize=(10, 6.6), layout="constrained")
    gs = fig.add_gridspec(2, 5, height_ratios=[1, 1.5])
    selection = []
    for i, (world, name) in enumerate(zip(worlds, NAMES + ["Pt/C"])):
        with h5py.File(ROOT / "data/simulated" / f"{world}.h5") as f:
            meta = f["metadata"].attrs
            p = f["ground_truth/atom_positions"][...]
            mask = f["ground_truth/defect_mask"][...]
            d = p[mask]
            size = float(meta["pixel_size_nm"])
            stride = int(meta["tile_size_px"] - meta["overlap_px"])
            pos = np.array([0.0, 0.0]) if i < 4 else p[0]
            (col, row) = np.clip(
                (pos / size / stride).astype(int),
                0,
                np.array(meta["grid_shape"])[::-1] - 1,
            )
            a = f[f"tiles/{row}_{col}"][...]
            ax = fig.add_subplot(gs[0, i])
            ax.imshow(a, cmap="gray", vmin=0, vmax=1, origin="upper")
            ax.axis("off")
            ax.set_title(name, fontweight="bold")
            bar = 32
            ax.plot([8, 8 + bar], [114, 114], color="white", lw=3)
            ax.text(8, 105, "32 pixels", color="white", fontsize=9)
            selection.append({"world": world, "row": int(row), "col": int(col)})
            if i == 0:
                overview = f["overview"][...]
    ax = fig.add_subplot(gs[1, :2])
    ax.imshow(overview, cmap="gray", vmin=0, vmax=1)
    ax.axis("off")
    ax.set_title("Release audit: stored SrTiO$_3$ overview", fontsize=10)
    ax.text(
        0.5,
        -0.06,
        "Repeated zero-intensity bands require assembly verification",
        transform=ax.transAxes,
        ha="center",
        fontsize=8,
    )
    ax = fig.add_subplot(gs[1, 2:])
    ax.axis("off")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    boxes = [
        (0.04, 0.67, "1  Navigate", "Choose an unvisited tile"),
        (0.56, 0.67, "2  Acquire", "Sample at a fixed local dose"),
        (0.56, 0.2, "3  Analyse", "Update accumulated findings"),
        (0.04, 0.2, "4  Evaluate", "Score every five acquisitions"),
    ]
    for x, y, title, sub in boxes:
        ax.add_patch(
            FancyBboxPatch(
                (x, y),
                0.4,
                0.22,
                boxstyle="round,pad=.015",
                facecolor="#edf3f6",
                edgecolor="#b8c9d3",
            )
        )
        ax.text(x + 0.2, y + 0.145, title, ha="center", weight="bold")
        ax.text(x + 0.2, y + 0.063, sub, ha="center", fontsize=8)
    for a, b in [
        ((0.455, 0.78), (0.535, 0.78)),
        ((0.76, 0.64), (0.76, 0.44)),
        ((0.535, 0.31), (0.455, 0.31)),
        ((0.24, 0.44), (0.24, 0.64)),
    ]:
        ax.annotate(
            "",
            xy=b,
            xytext=a,
            arrowprops={"arrowstyle": "->", "color": "#526876", "lw": 1.5},
        )
    ax.text(
        0.5,
        0.015,
        "Repeat until the acquisition budget is consumed",
        ha="center",
        fontsize=9,
    )
    fig.savefig(OUT / "figure1.pdf", bbox_inches="tight")
    plt.close(fig)
    (REPRO_OUT / "figure_sources.json").write_text(
        json.dumps({"figure1_tiles": selection}, indent=2) + "\n"
    )
    records = json.loads(
        (ROOT / "results/manuscript/release_validation.json").read_text()
    )["records"]
    rows = []
    material_order = ["srtio3", "batio3", "sige", "gan", "pt"]
    tex_names = ["\\ce{SrTiO3}", "\\ce{BaTiO3}", "SiGe", "GaN", "Pt/C"]
    for prefix, name in zip(material_order, tex_names):
        for level in ["easy", "medium", "hard"]:
            r = next(
                (
                    r
                    for r in records
                    if r["file"].split("/")[-1].startswith(prefix)
                    and r["file"].endswith(level + ".h5")
                )
            )
            m = r["metadata"]
            (fy, fx) = m["fov_nm"]
            status = "Core" if r["geometry_consistent"] else "Additional"
            rows.append(
                f"{name} & {level.capitalize()} & {r['tile_count']:,} & ${fy:.2f}\\times{fx:.2f}$ & {m['pixel_size_nm']:.2f} & {status} \\\\"
            )
    text = (
        "\\begin{table}[t]\n\\centering\n\\caption{Per-file release inventory. Field dimensions are $(y,x)$ in nm and pixel size is in nm. The 13 core records passing coordinate-bounds checks contain 13,600 tiles. The two additional hard records fail the bounds check and retain their original metadata. Physical calibration and image--label registration remain unverified for the release.}\n\\label{tab:release}\n\\small\n\\begin{tabular}{llrlll}\n\\toprule\nMaterial & Setting & Tiles & Declared field & Pixel size & Status \\\\\n\\midrule\n"
        + "\n".join(rows)
        + "\n\\bottomrule\n\\end{tabular}\n\\end{table}\n"
    )
    write_generated_table("supplementary.tex", "release_inventory_table", text)


def result_figures(results_path):
    episodes = json.loads(results_path.read_text())["episodes"]
    if not len(episodes) == 220:
        raise ValueError(f"Expected 220 completed episodes, found {len(episodes)}")

    def get(world, policy):
        e = sorted(
            [x for x in episodes if x["world"] == world and x["policy"] == policy],
            key=lambda x: x["seed"],
        )
        if not len(e) == 10:
            raise ValueError("Validation failed: len(e) == 10")
        return e

    def stats(world, policy):
        x = [e["dec_auc"] for e in get(world, policy)]
        return (np.mean(x), np.std(x, ddof=1))

    (fig, axes) = plt.subplots(1, 2, figsize=(10, 4), layout="constrained")
    for p, label, color in zip(POLICIES, LABELS, COLORS):
        e = get(WORLDS[0], p)
        curves = np.array([x["dec_scores"] for x in e])
        x = np.array(e[0]["dec_dose_fractions"])
        m = curves.mean(0)
        s = curves.std(0, ddof=1)
        axes[0].plot(x, m, label=label, color=color, lw=1.6)
        axes[0].fill_between(
            x, np.maximum(0, m - s), np.minimum(1, m + s), color=color, alpha=0.12
        )
    axes[0].set(
        xlabel="Fraction of acquisition budget",
        ylabel="Defect-census score",
        xlim=(0, 1),
        ylim=(0, 1),
        title="SrTiO$_3$: acquisition trajectories",
    )
    axes[0].legend(fontsize=7, loc="upper left", frameon=False)
    axes[0].grid(alpha=0.15)
    for i, d in enumerate(["easy", "medium", "hard"]):
        (m, s) = stats("batio3_interface_" + d, "raster_trained")
        axes[1].bar(i, m, color="#126aab", width=0.55, alpha=0.85)
        axes[1].errorbar(i, m, yerr=s, fmt="none", color="#283c4b", capsize=4)
    axes[1].set(
        xticks=range(3),
        xticklabels=["Easy", "Medium", "Hard"],
        ylabel="DEC-AUC",
        ylim=(0, 1),
        title="BaTiO$_3$: trained raster",
    )
    axes[1].grid(axis="y", alpha=0.15)
    fig.savefig(OUT / "figure3.pdf", bbox_inches="tight")
    plt.close(fig)
    rows = []
    for p, label in zip(POLICIES, LABELS):
        vals = [f"${m:.3f} \\pm {s:.3f}$" for (m, s) in [stats(w, p) for w in WORLDS]]
        rows.append(label + " & " + " & ".join(vals) + " \\\\")
    table = (
        "\\begin{table}[t]\n\\centering\n\\caption{Defect-census DEC-AUC under the shared 50-acquisition protocol: mean $\\pm$ sample standard deviation over ten seeds. Each column is one easy world.}\n\\label{tab:reference}\n\\small\n\\begin{tabular}{lcccc}\n\\toprule\nConfiguration & \\ce{SrTiO3} & \\ce{BaTiO3} & SiGe & GaN \\\\\n\\midrule\n"
        + "\n".join(rows)
        + "\n\\bottomrule\n\\end{tabular}\n\\end{table}\n"
    )
    write_generated_table("sec_experiments.tex", "reference_results_table", table)
    means = {w: {p: stats(w, p)[0] for p in POLICIES} for w in WORLDS}
    print(json.dumps(means, indent=2))
    (REPRO_OUT / "reference_summary.json").write_text(
        json.dumps(
            {
                w: {
                    p: {"mean": stats(w, p)[0], "sample_sd": stats(w, p)[1]}
                    for p in POLICIES
                }
                for w in WORLDS
            },
            indent=2,
        )
        + "\n"
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-only", action="store_true")
    parser.add_argument(
        "--results",
        type=Path,
        default=ROOT / "results/manuscript/reference_results.json",
    )
    parser.add_argument("--output-dir", type=Path, default=OUT)
    args = parser.parse_args()
    from stem_gym.artifacts import writable_output

    OUT = writable_output(args.output_dir, ROOT)
    REPRO_OUT = OUT / "reproducibility"
    REPRO_OUT.mkdir(parents=True, exist_ok=True)
    data_figures()
    if not args.data_only:
        result_figures(args.results)
