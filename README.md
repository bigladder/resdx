ResDX
=====

A python package for modeling residential direct expansion (DX) systems.

Purposes:

1. Comparing models
2. Developing new models (e.g., a unified model for RESNET software tools)
3. Calculating standard efficiency ratings, and performing inverse calculations of gross efficiencies from rated efficiencies:
    - SEER, EER, and HSPF (AHRI 210/240-2017)
    - SEER2, EER2, and HSPF2 (AHRI 210/240-2023)
    - CEER for room air conditioners and room heat pumps (DOE 10 CFR 430 Subpart B, Appendix F)
    - HEER for room heat pumps (ENERGY STAR Test Method to Determine Room Air Conditioner Heating Mode Performance, July 2024)
4. Generating ASHRAE 205 compliant representations

Development setup
-----------------

Requirements:

1. Python 3.8+
2. [Poetry](https://python-poetry.org/)

Project setup: `poetry install`

Testing: `poetry run doit`
