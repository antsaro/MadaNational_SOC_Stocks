# Madagascar National Soil Organic Carbon Stock Mapping (Update)

This repository contains the code used to fit the predictive models and generate the national
soil organic carbon (SOC) stock maps described in *"Update of the national soil carbon map of
Madagascar using additional data and earth observation platforms"*.

The study updates Madagascar's national topsoil (0–30 cm) SOC map and delivers the country's
first national subsoil (30–100 cm) and directly modelled full-profile (0–100 cm) SOC stock
products, each with pixel-level bootstrap uncertainty estimates.

## Study characteristics

- **Study area**: full national territory of Madagascar (587,000 km²)
- **SOC database**: 4,184 topsoil and 1,520 full-profile observations (2015–2025), combining
  plots from Ramifehiarivo et al. (2017) with new data from closed, open, agroecosystem and
  mangrove ecosystems
- **Depth intervals**: SOC₀₋₃₀, SOC₃₀₋₁₀₀, SOC₀₋₁₀₀
- **Covariates**: 10-layer stack (Sentinel-2 spectral indices, terrain attributes from AW3D30,
  WorldClim v2 climate surfaces, tree cover, ESA WorldCover 2020, national soil type), resampled
  to a common 30 m grid in Google Earth Engine
- **Models**: Random Forest and XGBoost, tuned with Optuna (TPE sampler) under a nested
  cross-validation scheme (10-fold outer, 4-fold inner)
- **Uncertainty**: non-parametric bootstrap (B = 100 replicates) producing per-pixel mean,
  standard deviation and coefficient of variation
- **Interpretation**: SHAP-based covariate importance and dependence analysis
- **Compute environment**: Google Earth Engine (covariate preprocessing) and Kaggle (free tier,
  dual NVIDIA T4 GPUs) for model training and spatial prediction, using RAPIDS cuML FIL for GPU
  inference

## Repository contents

This repository currently includes the code for:

- **Model fitting** - hyperparameter optimisation and training of the Random Forest and XGBoost
  models for each depth interval, under the repeated cross-validation and nested CV frameworks
  described in the manuscript
- **Spatial prediction and uncertainty mapping** - bootstrap-based national-scale prediction of
  SOC stocks and their associated standard deviation and coefficient of variation, at 30 m
  resolution

Covariate preprocessing (Google Earth Engine scripts) and manuscript figures/statistics are
maintained separately and are not part of this release.

## Example output

Mean predicted SOC stocks (a–c), prediction uncertainty as standard deviation (d–f), and
coefficient of variation (g–i) for the XGBoost model across the three depth intervals
(SOC₀₋₃₀, SOC₃₀₋₁₀₀, SOC₀₋₁₀₀):

![Mean, standard deviation and coefficient of variation of predicted SOC stocks across Madagascar for three depth intervals](figures/soc_stocks_uncertainty_map.png)

## Citation

If you use this code, please cite the associated manuscript (full reference to be added upon
publication).

## Contact

For questions about the code or the underlying database, please open an issue or contact the
corresponding author of the manuscript.
