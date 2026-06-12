"""Render the co-residence backend-RSS comparison chart for the README.

Reads slice-d.json from the skegdb.github.io repo, plots p50 backend
RSS in MiB across the corpus sweep (10K -> 1M) for each backend, and
saves an SVG + PNG into the repository's assets/ directory.

Inputs:
    SKEG_BENCH_JSON   path to slice-d.json (default: sibling repo path)
    SKEG_ASSETS_DIR   output directory      (default: repo assets/)

Outputs:
    coresidence-rss.svg
    coresidence-rss.png

Run from the repository root with the bench venv that has matplotlib
3.10 installed:

    /path/to/.venv-bench/bin/python3 assets/render-coresidence-rss.py
"""

import json
import os
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

HERE = Path(__file__).resolve().parent
# skeg/assets/ -> skeg/ -> skegdb/ -> skegdb.github.io/...
DEFAULT_JSON = (
    HERE.parent.parent / "skegdb.github.io" / "src" / "data" / "bench" / "slice-d.json"
)
JSON_PATH = Path(os.environ.get("SKEG_BENCH_JSON", DEFAULT_JSON))
OUT_DIR = Path(os.environ.get("SKEG_ASSETS_DIR", HERE))

BACKEND_STYLE = {
    "skeg": {"label": "skeg-pq128", "color": "#1f6feb", "marker": "o"},
    "qdrant": {"label": "qdrant (hnsw)", "color": "#f85149", "marker": "s"},
    "chroma": {"label": "chroma (hnsw)", "color": "#a371f7", "marker": "^"},
}


def load_series(path: Path):
    data = json.loads(path.read_text())
    series = {}
    for cell in data["cells"]:
        backend = cell["backend"]
        if backend not in BACKEND_STYLE:
            continue
        p50 = cell.get("backend_rss_mb_p50")
        rss_max = cell.get("backend_rss_mb_max", p50)
        if p50 is None:
            continue
        series.setdefault(backend, []).append(
            (cell["corpus_size"], p50, rss_max if rss_max is not None else p50)
        )
    for points in series.values():
        points.sort()
    return series


def render(series: dict, out_dir: Path) -> None:
    fig, ax = plt.subplots(figsize=(9, 4.8), dpi=160)
    fig.patch.set_facecolor("white")
    ax.set_facecolor("#fafafa")

    # Plot bands first so the median lines sit on top.
    for backend, points in series.items():
        style = BACKEND_STYLE[backend]
        xs = [p[0] for p in points]
        p50s = [p[1] for p in points]
        maxs = [p[2] for p in points]
        ax.fill_between(xs, p50s, maxs, color=style["color"], alpha=0.16, linewidth=0)
        ax.plot(
            xs,
            p50s,
            label=style["label"],
            color=style["color"],
            marker=style["marker"],
            markersize=6.5,
            linewidth=2.4,
        )

    ax.set_xscale("log")
    ax.set_xlabel("corpus size (vectors)")
    ax.set_ylabel("backend RSS (MiB)")
    ax.set_title(
        "Co-resident backend RSS while a 3B LLM serves RAG\n"
        "M1 Pro 16 GiB · Llama 3.2 Q4_K_M · mxbai-embed-large · line = p50, band = p50→max",
        fontsize=11,
        loc="left",
    )
    ax.grid(which="major", axis="y", color="#dddddd", linewidth=0.8)
    ax.grid(which="major", axis="x", color="#eeeeee", linewidth=0.6)
    ax.tick_params(axis="both", which="major", labelsize=9)

    def _human_mib(value, _pos):
        if value >= 1024:
            return f"{value / 1024:.1f} GiB"
        return f"{int(value)} MiB"

    def _human_count(value, _pos):
        if value >= 1_000_000:
            return f"{value / 1_000_000:.0f}M"
        if value >= 1_000:
            return f"{value / 1_000:.0f}K"
        return f"{int(value)}"

    ax.yaxis.set_major_formatter(mticker.FuncFormatter(_human_mib))
    ax.xaxis.set_major_formatter(mticker.FuncFormatter(_human_count))
    ax.legend(loc="upper left", frameon=False, fontsize=10)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    # At 1M the p50 lines visually converge because qdrant drops to
    # ~254 MiB while its max stays at 2.3 GiB. Spell both numbers out
    # at the right edge so the convergence is not mistaken for parity.
    one_m_skeg = next(
        ((p50, mx) for size, p50, mx in series.get("skeg", []) if size == 1_000_000),
        None,
    )
    one_m_qdr = next(
        ((p50, mx) for size, p50, mx in series.get("qdrant", []) if size == 1_000_000),
        None,
    )
    if one_m_skeg is not None:
        p50, mx = one_m_skeg
        ax.annotate(
            f"skeg @1M\np50 {int(p50)} MiB · max {int(mx)} MiB",
            xy=(1_000_000, mx),
            xytext=(8, 28),
            textcoords="offset points",
            fontsize=9,
            color="#1f6feb",
        )
    if one_m_qdr is not None:
        p50, mx = one_m_qdr
        gib_max = mx / 1024
        ax.annotate(
            f"qdrant @1M\np50 {int(p50)} MiB · max {gib_max:.1f} GiB",
            xy=(1_000_000, mx),
            xytext=(-170, -14),
            textcoords="offset points",
            fontsize=9,
            color="#f85149",
        )

    fig.tight_layout()

    # Explicit ratio caption placed under the axes so it never
    # overlaps the data. The visual convergence of the p50 lines at
    # 1M is a measurement quirk, not parity.
    if one_m_skeg is not None and one_m_qdr is not None:
        p50_ratio = one_m_qdr[0] / one_m_skeg[0]
        max_ratio = one_m_qdr[1] / one_m_skeg[1]
        fig.subplots_adjust(bottom=0.20)
        fig.text(
            0.5,
            0.02,
            f"@1M corpus: qdrant/skeg p50 ≈ {p50_ratio:.1f}×  ·  max ≈ {max_ratio:.0f}×  "
            "(p50 lines look close, max diverges by a full order of magnitude)",
            ha="center",
            va="bottom",
            fontsize=9,
            color="#444444",
        )

    svg_path = out_dir / "coresidence-rss.svg"
    png_path = out_dir / "coresidence-rss.png"
    fig.savefig(svg_path, format="svg")
    fig.savefig(png_path, format="png")
    print(f"wrote {svg_path}")
    print(f"wrote {png_path}")


def main() -> None:
    series = load_series(JSON_PATH)
    if not series:
        raise SystemExit(f"no backend RSS data found in {JSON_PATH}")
    render(series, OUT_DIR)


if __name__ == "__main__":
    main()
