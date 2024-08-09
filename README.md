This repo contains the codebase for the manuscript:

## Quantitative Modeling of the Emergence of Macroscopic Grid-like Representations

**Authors:** Ikhwan Bin Khalid, Eric T. Reifenstein, Naomi Auer, Lukas Kunz, Richard Kempter  

**Publication:** eLife2024;13:e85742

**DOI:** https://doi.org/10.7554/eLife.85742

---
## How to Use the Scripts


### 1. Generate Manuscript Figures
**Files:** `.py` files starting with `"Produce_"`  

**Purpose:** These files will compile the figures for the manuscript.


### 2. Required Subroutines
Some `"Produce_"` `.py` files require `"Subroutine_"` scripts to be run first to generate the required data. In the case of the clustering figure, the subroutines generate '.png' files which were combined with the main figure using Inkscape. Below is a list of these files and their required subroutines:

| **Script**                       | **Subroutine(s)**                                                                         |
|----------------------------------|-------------------------------------------------------------------------------------------|
| Produce_mainfigure_conjunctive.py    | Subroutine_mainfigure_conjunctive_paramsearch.py |
| Produce_mainfigure_repsupp.py | Subroutine_mainfigure_repsupp_paramsearch.py |
| Produce_mainfigure_clustering.py | Subroutine_mainfigure_clustering_panel_T.py <br>Subroutine_mainfigure_clustering_randfield.py |
| Produce_mainfigure_summary.py | Subroutine_mainfigure_summary_data.py |
| Produce_suppfigure_boundaries.py | Subroutine_suppfigure_boundaries_rotate.py <br>Subroutine_suppfigure_boundaries_size.py |


### 3. Miscellaneous Scripts
**Files:** `.py` files starting with `"misc_"` 

**Purpose:** These scripts are not used for the figures in the manuscript. They generate additional data or figures for model exploration.


### 4. Settings

**Files:** `settings.py` 

**Purpose:** This file contains hyperparameters for all scripts, including grid cell parameters.

---

## Contact

For any questions, please contact:  

**Email:** ikhwankhalid92@gmail.com