# LFMC_from_MODIS
Live fuel moisture content estimation from MODIS: a deep learning approach

This code reproduces the results in the paper. It uses Google Earth Engine as the data source, which saves downloading and pre-processing large volumes of MODIS reflectance data. Due to this, and different random variable seed settings, the results will differ slightly from those shown in the paper, but should be broadly similar.

## Data
Four data sources are used:
1. The `Globe-LFMC.xlsx` datatset
2. The NASA SRTM Digital Elevation 30m data. Google Earth Engine product: USGS/SRTMGL1_003 - https://developers.google.com/earth-engine/datasets/catalog/USGS_SRTMGL1_003
3. MODIS MCD43A4.006 Nadir BRDF-Adjusted Reflectance Daily 500m data. Google Earth Engine product: MODIS/006/MCD43A4 - https://developers.google.com/earth-engine/datasets/catalog/MODIS_006_MCD43A4
4. MODIS MOD10A1.006 Terra Snow Cover Daily Global 500m data. Google Earth Engine product: MODIS/006/MCD10A1 - https://developers.google.com/earth-engine/datasets/catalog/MODIS_006_MOD10A1

## Requirements
1. Python 3.8 with packages listed in [requirements.txt](../requirements.txt). The code was written and tested using Anaconda. [InstallNotes.md](../InstallNotes.md) contains some notes about how to set up a suitable Anaconda virtual environment.
2. An authenticated Google Earth Engine account
3. A copy of the `Globe-LFMC.xlsx` dataset

## Code
The iPython notebooks can be used to extract data from Google Earth Engine and run the scenarios from the paper. Steps are:
1. Run the `Extract Auxiliary Data.ipynb` notebook to pre-process the Globe-LFMC dataset and download the SRTM DEM data.
2. Run the `Extract MODIS Data.ipynb` notebook to download the MODIS reflectance data and remove the samples collected under snow conditions.
3. Run the three scenarios: `LFMC Scenario A.ipynb`, `LFMC Scenario B.ipynb`, and `LFMC Scenario C.ipynb`. These can be run in any order.
4. Run the `Figure 3 plots.ipynb` notebook to produce a figure summarising all scenario results. This produces a plot similar to Fig. 3 in the paper

## Contributors
1. Lynn Miller: https://github.com/lynn-miller
2. Luijun Zhu: https://github.com/rszlj

## Acknowledgements

### Globe-LFMC dataset
Yebra, M., Scortechini, G., Badi, A., Beget, M. E., Boer, M. M., Bradstock, R. A., Chuvieco, E., Danson, F. M., Dennison, P. E., Resco de Dios, V., Di Bella, C. M., Forsyth, G., Frost, P., García, M., Hamdi, A., He, B., Jolly, M., Kraaij, T., Martín, M. P., … Ustin, S. (2019). Globe-LFMC, a global plant water status database for vegetation ecophysiology and wildfire applications. Scientific Data, 6(1), 155. https://doi.org/10.1038/s41597-019-0164-9

### Google Earth Engine Data
1. Google Earth Engine: Gorelick, N., Hancher, M., Dixon, M., Ilyushchenko, S., Thau, D., & Moore, R. (2017). Google Earth Engine: Planetary-scale geospatial analysis for everyone. Remote Sensing of Environment.
2. NASA SRTM Digital Elevation 30m dataset: Farr, T.G., Rosen, P.A., Caro, E., Crippen, R., Duren, R., Hensley, S., Kobrick, M., Paller, M., Rodriguez, E., Roth, L., Seal, D., Shaffer, S., Shimada, J., Umland, J., Werner, M., Oskin, M., Burbank, D., and Alsdorf, D.E. (2007). The shuttle radar topography mission: Reviews of Geophysics, v. 45, no. 2, RG2004, at https://doi.org/10.1029/2005RG000183.
3. MODIS MCD43A4 Version 6 dataset: Schaaf, C., Wang, Z. (2015). <i>MCD43A4 MODIS/Terra+Aqua BRDF/Albedo Nadir BRDF Adjusted Ref Daily L3 Global - 500m V006</i> [Data set]. NASA EOSDIS Land Processes DAAC. Accessed 2021-03-19 from https://doi.org/10.5067/MODIS/MCD43A4.006 
4. MODIS MOD10A1 Version 6 dataset: Hall, D. K., Salomonson, V. V., and Riggs, G. A. (2016). MODIS/Terra Snow Cover Daily L3 Global 500m Grid. Version 6. Boulder, Colorado USA: NASA National Snow and Ice Data Center Distributed Active Archive Center.

### Temp_CNN model
Pelletier, C., Webb, G. I., & Petitjean, F. (2019). Temporal convolutional neural network for the classification of satellite image time series. Remote Sensing, 11(5), 1–25. https://doi.org/10.3390/rs11050523

### Main Python Packages
1. TensorFlow: Abadi, M., Agarwal, A., Barham, P., Brevdo, E., Chen, Z., Citro, C., Corrado, G. S., Davis, A., Dean, J., Devin, M., Ghemawat, S., Goodfellow, I., Harp, A., Irving, G., Isard, M., Jia, Y., Jozefowicz, R., Kaiser, L., Kudlur, M., … Research, G. (2015). TensorFlow: Large-Scale Machine Learning on Heterogeneous Distributed Systems. www.tensorflow.org.
2. Keras: Chollet, F., & others. (2015). Keras. GitHub. Retrieved from https://github.com/fchollet/keras
3. Numpy: Harris, C. R., Millman, K. J., van der Walt, S. J., Gommers, R., Virtanen, P., Cournapeau, D., Wieser, E., Taylor, J., Berg, S., Smith, N. J., Kern, R., Picus, M., Hoyer, S., van Kerkwijk, M. H., Brett, M., Haldane, A., del Río, J. F., Wiebe, M., Peterson, P., … Oliphant, T. E. (2020). Array programming with NumPy. Nature, 585(7825), 357–362. https://doi.org/10.1038/s41586-020-2649-2 
4. Pandas: Reback, J., McKinney, W., jbrockmendel, Bossche, J. Van den, Augspurger, T., Cloud, P., gfyoung, Hawkins, S., Sinhrks, Roeschke, M., Klein, A., Petersen, T., Tratner, J., She, C., Ayd, W., Naveh, S., Garcia, M., patrick, Schendel, J., … h-vetinari. (2021). pandas-dev/pandas: Pandas 1.2.1. https://doi.org/10.5281/zenodo.4452601
5. Scipy: Virtanen, P., Gommers, R., Oliphant, T. E., Haberland, M., Reddy, T., Cournapeau, D., Burovski, E., Peterson, P., Weckesser, W., Bright, J., van der Walt, S. J., Brett, M., Wilson, J., Millman, K. J., Mayorov, N., Nelson, A. R. J., Jones, E., Kern, R., Larson, E., … Vázquez-Baeza, Y. (2020). SciPy 1.0: fundamental algorithms for scientific computing in Python. Nature Methods, 17(3), 261–272. https://doi.org/10.1038/s41592-019-0686-2
6. Scikit-Learn: Pedregosa, F., Michel, V., Grisel, O., Blondel, M., Prettenhofer, P., Weiss, R., Dubourg, V., Vanderplas, J., Passos, A., Cournapeau, D., Brucher, M., Perrot, M., & Duchesnay, É. (2011). Scikit-learn: Machine Learning in Python
7. Matplotlib: Hunter, J. D. (2007). Matplotlib: A 2D Graphics Environment. In Computing in Science & Engineering (Vol. 9, Issue 3, pp. 90–95). IEEE Computer Society. https://doi.org/10.1109/MCSE.2007.55
