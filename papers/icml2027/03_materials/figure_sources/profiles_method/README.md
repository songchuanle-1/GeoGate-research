# Figure 1 and Figure 2 editorial sources

This folder rebuilds the two revised figures from the unchanged October 26
frozen figure evidence. It does not run a model, rescore predictions, or
recompute a statistic.

Run `python regenerate.py` from this folder with Python, NumPy, and Matplotlib.
The existing Times New Roman files are loaded from `C:/Windows/Fonts`.
The command writes PDF/SVG/600-dpi PNG exports in `exports/` and copies the two
PDFs to the parent manuscript folder.

The frozen CSV and render-evidence JSON files are included in `frozen/`.
`verification.json` records the baseline comparison of plotted coordinates,
patch geometry, numerical evidence, font constraints, and other unchanged
figure PDFs. `pdf_text_audit.json` records PDF page boxes and numeric text.
Figure 1's numeric text is unchanged. Figure 2 adds only `L0-14` to explain
that prompt-cache compaction also includes the early layers.

`common.py`, `profiles.py`, and `architecture.py` are derived from the
October 26 conference redraw source. Changes are limited to descriptive
attention-region labels and the explicit prompt-cache-compaction description.
All 192 attention-profile values, layer boundaries, canvas sizes, axes,
plotted lines, and schematic token geometry are preserved.
