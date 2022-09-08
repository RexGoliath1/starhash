Hello World!


## Generating Hipparcos
Per ESA Tetra documentation:
https://github.com/esa/tetra3
https://tetra3.readthedocs.io/en/latest/api_docs.html#tetra3-a-fast-lost-in-space-plate-solver-for-star-trackers
"‘hip_main’ and ‘tyc_main’ are available from <https://cdsarc.u-strasbg.fr/ftp/cats/I/239/>"

COTS Star Tracker repo used newer 311 version (Major cleanups):
https://cdsarc.cds.unistra.fr/viz-bin/cat/I/311
https://www.aanda.org/articles/aa/pdf/2007/41/aa8357-07.pdf

Steps:
1. Query with Vizier:
https://vizier.cfa.harvard.edu/viz-bin/VizieR-3?-source=I/311/hip2
2. Preferences (Left column): HTML Table dropdown --> Tab-Seperated-Values

This mimics COTS Star Tracker outputs. Unclear why pm generated values were unused.


