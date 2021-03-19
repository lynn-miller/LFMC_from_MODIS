# LFMC_from_MODIS
Live fuel moisture content estimation from MODIS: a deep learning approach

This code reproduces the results in the paper. It uses Google Earth Engine as the data source, which saves downloading and pre-processing large volumes of MODIS reflectance data. Due to this, and different random variable seed settings, the results will differ slightly from those shown in the paper, but should be broadly similar.

## Data

## Requirements
1. Python with packages listed in [requirements.txt](../requirements.txt)
2. An authenticated Google Earth Engine account
3. A copy of the Globe-LFMC.xlsx dataset

## Code
The iPython notebooks can be used to extract data from Google Earth Engine and run the scenarios from the paper. Steps are:
1. Run the `Extract Auxiliary Data.ipynb` notebook to pre-process the Globe-LFMC dataset and download the SRTM DEM data.
2. Run the `Extract MODIS Data.ipynb` notebook to download the MODIS reflectance data and remove the samples collected under snow conditions.
3. Run the three scenarios: `LFMC Scenario A.ipynb`, `LFMC Scenario B.ipynb`, and `LFMC Scenario C.ipynb`. These can be run in any order.
4. Run the `Figure 3 plots.ipynb` notebook to produce a figure summarising all scenario results. This produces a plot similar to Fig. 3 in the paper

## Contributors

## Acknowledgements

### Globe-LFMC dataset

### Google Earth Engine Data

### Temp_CNN model

### Main Python Packages
