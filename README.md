This repo contains the code for the manuscript "Quantitative modeling of the emergence of macroscopic grid-like representations".

Ikhwan Bin Khalid, Eric T. Reifenstein, Naomi Auer, Lukas Kunz, Richard Kempter. - Quantitative modeling of the emergence of macroscopic grid-like representations. bioRxiv 2022.12.20.521210; doi: https://doi.org/10.1101/2022.12.20.521210

---
#### How to use the scripts:

##### 1. ".py" files starting with "Produce" will compile the figures of the manuscript.

##### 2. Some "Produce" .py files require "Subroutine" scripts to be run first. These are:

| **Script**                       | **Subroutine(s)**                                                                         |
|----------------------------------|-------------------------------------------------------------------------------------------|
| Produce_mainfigure_clustering.py | Subroutine_mainfigure_clustering_panel_T.py <br>Subroutine_mainfigure_clustering_randfield.py |
| Produce_mainfigure_repsupp.py    | Subroutine_mainfigure_repsupp_paramsearch.py                                              |
| Produce_mainfigure_summary.py    | Subroutine_mainfigure_summary_data.py                                                     |
| Produce_suppfigure_boundaries.py | Subroutine_suppfigure_boundaries_rotate.py <br>Subroutine_suppfigure_boundaries_size.py       |

##### 3. ".py" files starting with "misc" are not used in the figures of the manuscript. They are used to generate additional data or figures for the purpose of exploring the model.

---
Please contact me if you have any questions:
ikhwankhalid92@gmail.com