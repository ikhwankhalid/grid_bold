This repo contains the code for the manuscript:


## Quantitative Modeling of the Emergence of Macroscopic Grid-like Representations



**Authors:** Ikhwan Bin Khalid, Eric T. Reifenstein, Naomi Auer, Lukas Kunz, Richard Kempter  

**Publication:** bioRxiv 2022.12.20.521210  

**DOI:** [10.1101/2022.12.20.521210](https://doi.org/10.1101/2022.12.20.521210)



---



## How to Use the Scripts



### 1. Generate Manuscript Figures



**Files:** `.py` files starting with "Produce"  

**Purpose:** These files will compile the figures for the manuscript.



### 2. Required Subroutines



Some "Produce" `.py` files require subroutine scripts to be run first. Below is a list of these files and their required subroutines:



| **Script**                       | **Subroutine(s)**                                                                         |
|----------------------------------|-------------------------------------------------------------------------------------------|
| Produce_mainfigure_clustering.py | Subroutine_mainfigure_clustering_panel_T.py <br>Subroutine_mainfigure_clustering_randfield.py |
| Produce_mainfigure_repsupp.py    | Subroutine_mainfigure_repsupp_paramsearch.py                                              |
| Produce_mainfigure_summary.py    | Subroutine_mainfigure_summary_data.py                                                     |
| Produce_suppfigure_boundaries.py | Subroutine_suppfigure_boundaries_rotate.py <br>Subroutine_suppfigure_boundaries_size.py       |



### 3. Miscellaneous Scripts



**Files:** `.py` files starting with "misc"  

**Purpose:** These scripts are not used for the figures in the manuscript. They generate additional data or figures for model exploration.



### 4. Settings

**Files:** `settings.py` 

**Purpose:** This file contains hyperparameters for all scripts, including grid cell parameters.

---



## Contact



For any questions, please contact:  

**Email:** ikhwankhalid92@gmail.com