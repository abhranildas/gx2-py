#!/usr/bin/env python
"""Regenerate the getting-started assets for gx2-py from GettingStarted.ipynb.

Executes the notebook fresh (so its outputs -- including every plot --
reflect the current code), saves the executed notebook back in place, then:

1. Extracts every image/png cell output into getting-started/ (the folder
   is cleared first), named and captioned via FIGURE_MANIFEST below.
2. Rewrites the auto-generated block of README.md (between the
   BEGIN/END GENERATED markers in the "Examples" section) from the
   notebook's cells: markdown cells (heading demoted one level), code cells
   (as ```python blocks), and their text/image outputs.

Run this before tagging a release, whenever GettingStarted.ipynb changes:

    python scripts/build_getting_started.py

Pass --no-execute to skip re-running the notebook and just re-export from
whatever outputs are currently saved in it (faster, for iterating on this
script's formatting).

CI (.github/workflows/publish.yml) also runs this on every release, so the
published README/images always reflect a fresh run of the notebook.
"""
from __future__ import annotations

import argparse
import base64
import copy
import re
import sys
from pathlib import Path

import nbformat
from nbclient import NotebookClient

REPO = Path(__file__).resolve().parent.parent
NOTEBOOK_PATH = REPO / "GettingStarted.ipynb"
IMAGES_DIR = REPO / "getting-started"
README_PATH = REPO / "README.md"

# Images are embedded via absolute raw.githubusercontent.com URLs (matching
# the logo at the top of the README) rather than repo-relative paths, since
# PyPI renders the README's long_description with no base URL to resolve
# relative links against -- a relative path renders fine on GitHub but 404s
# on the PyPI project page.
RAW_BASE = "https://raw.githubusercontent.com/abhranildas/gx2-py/main"

# Executed in a temporary cell prepended to the notebook (then discarded
# before saving) so plots are captured as image/png outputs regardless of
# the kernel's default matplotlib backend.
MATPLOTLIB_SETUP = "%matplotlib inline"

BEGIN_MARKER = (
    "<!-- BEGIN GENERATED: getting-started "
    "(do not edit by hand; regenerate with `python scripts/build_getting_started.py`) -->"
)
END_MARKER = "<!-- END GENERATED: getting-started -->"

# Maps a code cell's persistent notebook id -> (filename stem, alt text) for
# each cell whose output is a plot. Add an entry here when you add a new
# plotting cell to the notebook; cells producing an image with no entry get
# an auto-generated placeholder name/caption instead (a warning is printed)
# so you can fill it in.
FIGURE_MANIFEST = {
    "df38957d": ("01_sample_histogram",
                 "Histogram of samples, with the expected mode marked"),
    "162476c7": ("02_pdf_vs_histogram",
                 "Computed PDF overlaid on the sampled histogram"),
    "c71d2ba1": ("03_cdf_vs_histogram",
                 "Computed CDF overlaid on the sampled cumulative histogram"),
    "892da8b1": ("04_cdf_methods_nonelliptic",
                 "CDF from the IFFT, ray and Imhof methods, overlaid"),
    "7748048f": ("05_pdf_methods_nonelliptic",
                 "PDF from the IFFT, ray and Imhof methods, overlaid"),
    "5a04915a": ("06_cdf_methods_elliptic",
                 "CDF from the IFFT, Ruben, ray and Imhof methods, overlaid"),
    "eee2769f": ("07_pdf_methods_elliptic",
                 "PDF from the IFFT, Ruben, ray and Imhof methods, overlaid"),
    "50d966b6": ("08_cdf_infinite_lower_tail",
                 "log10(CDF) in the lower tail, from the IFFT, tail, Pearson, "
                 "ray and Imhof methods"),
    "fb61557f": ("09_pdf_infinite_upper_tail",
                 "log10(PDF) in the upper tail, from the IFFT, tail, Pearson, "
                 "ray and Imhof methods"),
    "cceb0d45": ("10_cdf_finite_lower_tail",
                 "log10(CDF) in a finite lower tail, from the IFFT, ellipse, "
                 "Pearson, ray, Imhof and Ruben methods"),
    "5f07f32f": ("11_normal_scatter", "Scatter of the sampled normal vectors"),
    "053ecb78": ("12_quadform_pdf_vs_histogram",
                 "Computed PDF of q overlaid on its sampled histogram"),
    "de5b870d": ("13_characteristic_function",
                 "Characteristic function traced in the complex plane"),
    "a563cc74": ("14_taylor_native",
                 "True cdf vs. its 2nd-order Taylor approximation, as "
                 "lambda_1 is varied"),
    "c37e0886": ("15_taylor_boundary",
                 "True cdf vs. its 2nd-order Taylor approximation, as "
                 "Q_2(1,1) is varied"),
    "f5a6b7c8": ("16_optimal_boundary",
                 "The two classes' covariance ellipses and the optimal "
                 "quadratic boundary between them"),
}


def execute_notebook(nb):
    """Run every cell and return a copy of nb with fresh outputs.

    A temporary "%matplotlib inline" cell is prepended for the run (and
    dropped again before returning) so plots are captured as image/png
    outputs no matter what backend the kernel would otherwise default to.
    """
    exec_nb = copy.deepcopy(nb)
    exec_nb.cells.insert(0, nbformat.v4.new_code_cell(MATPLOTLIB_SETUP))
    client = NotebookClient(
        exec_nb,
        timeout=1200,
        kernel_name="python3",
        resources={"metadata": {"path": str(REPO)}},
    )
    client.execute()
    exec_nb.cells.pop(0)
    return exec_nb


def demote_headings(markdown: str) -> str:
    """Shift every markdown heading in this cell down one level (## -> ###)."""
    return re.sub(r"^(#+)(\s)", r"#\1\2", markdown, flags=re.MULTILINE)


def image_outputs(cell):
    for out in cell.get("outputs", []):
        if "image/png" in out.get("data", {}):
            yield out


def text_output(cell) -> str:
    """Concatenate this cell's stdout/repr outputs into one plain-text blob."""
    parts = []
    for out in cell.get("outputs", []):
        if out.get("output_type") == "stream":
            text = out.get("text", "")
        elif out.get("output_type") == "execute_result":
            text = out.get("data", {}).get("text/plain", "")
        else:
            continue
        parts.append("".join(text) if isinstance(text, list) else text)
    return "".join(parts).rstrip("\n")


def build_examples_markdown(nb) -> tuple[str, set[str]]:
    lines = []
    used_files = set()
    fig_counter = 0
    # cell 0 is the intro (title, help() list, citation) -- the README
    # already covers that in its own header, so start from cell 1.
    for cell in nb.cells[1:]:
        if cell.cell_type == "markdown":
            lines.append(demote_headings(cell.source.rstrip()))
            lines.append("")
        elif cell.cell_type == "code":
            src = cell.source.rstrip()
            if src:
                lines.append("```python")
                lines.append(src)
                lines.append("```")

            imgs = list(image_outputs(cell))
            if imgs:
                for out in imgs:
                    fig_counter += 1
                    cell_id = cell.get("id", "")
                    stem, alt = FIGURE_MANIFEST.get(cell_id, (
                        f"fig_{fig_counter:02d}",
                        "Plot output (add a FIGURE_MANIFEST entry in "
                        "scripts/build_getting_started.py for a better name/caption)",
                    ))
                    if cell_id not in FIGURE_MANIFEST:
                        print(
                            f"warning: cell {cell_id!r} has no FIGURE_MANIFEST "
                            f"entry; using placeholder {stem!r}",
                            file=sys.stderr,
                        )
                    filename = f"{stem}.png"
                    used_files.add(filename)
                    png_bytes = base64.b64decode(out["data"]["image/png"])
                    (IMAGES_DIR / filename).write_bytes(png_bytes)
                    lines.append(f"![{alt}]({RAW_BASE}/getting-started/{filename})")
            else:
                text = text_output(cell)
                if text:
                    lines.append("```")
                    lines.append(text)
                    lines.append("```")
            lines.append("")
    return "\n".join(lines).rstrip() + "\n", used_files


def update_readme(examples_markdown: str) -> None:
    readme = README_PATH.read_text(encoding="utf-8")
    pattern = re.compile(re.escape(BEGIN_MARKER) + r".*?" + re.escape(END_MARKER), re.DOTALL)
    if not pattern.search(readme):
        raise SystemExit(
            "README.md is missing the generated-block markers.\n"
            f"Add:\n{BEGIN_MARKER}\n{END_MARKER}\n"
            "around the worked-examples section first."
        )
    replacement = f"{BEGIN_MARKER}\n\n{examples_markdown}\n{END_MARKER}"
    # a plain string replacement (not pattern.sub(replacement, ...)) so that
    # backslashes in the markdown (e.g. LaTeX like \lambda) aren't
    # interpreted as regex backreferences
    README_PATH.write_text(pattern.sub(lambda _: replacement, readme), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--no-execute", action="store_true",
        help="skip re-running the notebook; export from its currently saved outputs",
    )
    args = parser.parse_args()

    IMAGES_DIR.mkdir(exist_ok=True)
    # start from a clean folder each run, so a renamed/removed plot cell
    # can't leave a stale, no-longer-referenced image behind
    for png in IMAGES_DIR.glob("*.png"):
        png.unlink()

    nb = nbformat.read(NOTEBOOK_PATH, as_version=4)
    if not args.no_execute:
        nb = execute_notebook(nb)
        nbformat.write(nb, NOTEBOOK_PATH)

    examples_markdown, used_files = build_examples_markdown(nb)
    update_readme(examples_markdown)
    print(
        f"Updated {README_PATH.relative_to(REPO)} and {len(used_files)} "
        f"image(s) in {IMAGES_DIR.relative_to(REPO)}/"
    )


if __name__ == "__main__":
    main()
