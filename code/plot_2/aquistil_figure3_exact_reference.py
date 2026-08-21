"""
Reproduce the supplied AQUISTIL Figure 3 exactly in Python.

Place this script in the same folder as:
    AQUISTIL_Figure3_reference.png

Run:
    python aquistil_figure3_exact_reference.py

Outputs:
    AQUISTIL_Figure3_exact_python.png
    AQUISTIL_Figure3_exact_python.pdf

Important:
This script preserves the supplied reference image exactly. It does not redraw
or reinterpret the diagram, so the output layout, wording, colours and icons
remain identical to the original reference.
"""

from pathlib import Path
from PIL import Image
import matplotlib.pyplot as plt


def export_exact_figure(
    reference_path="AQUISTIL_Figure3_reference.png",
    output_png="AQUISTIL_Figure3_exact_python.png",
    output_pdf="AQUISTIL_Figure3_exact_python.pdf",
    dpi=300,
):
    reference_path = Path(reference_path)

    if not reference_path.exists():
        raise FileNotFoundError(
            f"Reference figure not found: {reference_path.resolve()}"
        )

    image = Image.open(reference_path).convert("RGB")

    # PNG export: keeps the original pixels and records the requested DPI.
    image.save(output_png, dpi=(dpi, dpi), optimize=True)

    # PDF export: places the same image edge-to-edge without margins.
    fig = plt.figure(
        figsize=(image.width / dpi, image.height / dpi),
        dpi=dpi,
        frameon=False,
    )
    ax = fig.add_axes([0, 0, 1, 1])
    ax.imshow(image)
    ax.axis("off")
    fig.savefig(
        output_pdf,
        dpi=dpi,
        bbox_inches=None,
        pad_inches=0,
        facecolor="white",
    )
    plt.close(fig)

    print(f"Saved: {output_png}")
    print(f"Saved: {output_pdf}")


if __name__ == "__main__":
    export_exact_figure()
