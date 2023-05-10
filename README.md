# StarHash
This repository contains the star identification and attitude determination algorithm presented in the paper below:

[TETRA: Star Identification with Hash Tables](https://digitalcommons.usu.edu/cgi/viewcontent.cgi?article=3655&context=smallsat)

This paper builds on popular methods such as Triangle/Pyramid/ISA approaches by Mortari, Salmaan, and Christan. Paper is implemented in C by author and in Python by ESA. This method was used by Pedrotty and co for Seeker / R5 calibration alogirthm validation, but not flown (they used a classic triangle / K-vector method written by J. Christan's group). This repo implements this paper in C++ using the OpenCV, Eigen, and Ceres libraries. Catalog generation only currently.

Purpose:
- General Attitude estimation through star locations
- Allows for mostly sensor agnostic approach (CCD,CMOS, IR, Neuromorphic, Multi-Camera)
- <b>Fast</b> lost-in-space initialization
- Low real time compute
- Capable of using multiple catalogs
- Agnostic to number of stars (star patterns can be 4+ depending on desired capabilities)

Tradeoffs:
- Somewhat larger memory requirements (hash table and general catalog info required)
- High compute required to create catalog
- Assumption: Catalogs of star positions and ISA's (Intra Star Angles) are stored on rad hard EE-PROM type devices.
- If ISA catalogs are lost from memory, options for recompiling must limited. Would take on scale of minutes to re-initialize.

Challenges: 
- Hash does not neccessarily capture star pattern density in local region. 
- Initial star candidates in catalog are brightest stars, must limit nearby stars according to camera iFOV (instantaneous (pixel) field of view). 

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

## Generating GAIA/UCAC4/Tycho
Not implemented. Just researching.

Amazing repos here (generally a really cool developer, this guy figured out the chinese rocket lunar collision):

https://github.com/Bill-Gray/star_cats

Need to implement some of these. Other possibilities:

https://github.com/Nerenisa/UCAC4-UCAC5

https://github.com/jobovy/gaia_tools


Good tooling in astroquery, but a lot of work needed for cleaning up:
https://arxiv.org/pdf/1804.09366.pdf


## State Diagrams
TODO: Implement in plantUML or equivilent. 

1. Catalog Generation
2. Lost In Space Phase (LISP)
3. Tracking Optical-Flow Phase (TOP)

## Running in NASA GIANT
This will simulate the camera optics (generally) and render star field images based on a number of star catalogs and ephemeris data.

I have this running in docker to generate images:
https://github.com/nasa/giant

Warning: There are a few tricks with OpenGL to force a headless container to render, need to write this up.

More documentation here:
https://aliounis.github.io/giant_documentation/

## Validation
Plan to validate using similar approach as Pedrotty against [Astronometry.net algorithm (ISA Triangle)](https://arxiv.org/pdf/0910.2233.pdf).

Cool discussion here:
https://astronomy.stackexchange.com/questions/33575/how-the-heck-does-astrometry-net-work

Some python implemented in the COTS repo:
https://github.com/nasa/COTS-Star-Tracker/blob/master/py_src/tools/astrometry_output_comparison.py