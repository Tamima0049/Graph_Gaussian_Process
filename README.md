# GNN-GP: Graph-based soil moisture imputation with uncertainty (Cook Agronomy Farm)

Code for the Cook Agronomy Farm experiments in:

> T. Rashid, A. Majumder, B. Natarajan (2026). **Graph-based spatio-temporal soil moisture imputation with uncertainty quantification.** *Smart Agricultural Technology* 15, 102455. https://doi.org/10.1016/j.atech.2026.102455

GNN-GP fills in volumetric water content (VWC) at locations without sensors. A GraphSAGE network turns each sensor's position (x, y, depth) into an embedding using its neighbours on a k-nearest-neighbour graph. A Gaussian process (GP) then uses that embedding, together with the date and crop type, to predict VWC with a mean and an uncertainty. The GNN and GP are trained together.

## Quick start

```bash
pip install -r requirements.txt     # install PyTorch for your platform first
python gnngp_cook.py
```

This runs Case 2 of the paper. 30, 50 and 70 % of the sensor locations are held out in turn, using seeds 42 and 43. The script prints MAE, RMSE, PICP and NMPIW for each run and the mean over the two seeds. It uses a GPU if one is available. The full experiment takes about 30 seconds on a GPU and about 45 minutes on a CPU.

Options:

| Flag | Default | Meaning |
|---|---|---|
| `--data` | `data/cook_2015_springsummer_final.csv` | input CSV |
| `--missing` | `0.3 0.5 0.7` | fractions of sensor locations held out |
| `--seeds` | `42 43` | random seeds |
| `--epochs` | `60` | training epochs |
| `--dropout` | `0.2` | dropout after the hidden GraphSAGE layer |
| `--device` | `cuda` if available, else `cpu` | where to run |

## Model

| Part | Setting (paper Table 2) |
|---|---|
| Graph | k-NN, k = 5, on scaled (x, y, depth); binary edges. Training points link to training points; test points link only to training points. |
| GNN | 2 GraphSAGE layers, mean aggregation, 8 hidden / output units, tanh, 20 % dropout |
| GP kernel | `alpha * K_crop * K_spatial * K_temporal` |
| `K_spatial` | RBF with 8 length-scales (ARD) on the GNN embedding |
| `K_temporal` | RBF with 1 length-scale on the scaled date |
| `K_crop` | IndexKernel over the 5 crops, rank 1 (Cook only: mixed-cropping field) |
| GP mean | constant, learned |
| Training | GNN and GP parameters optimised together with Adam, lr 0.01, 60 epochs, full batch, maximising the GP marginal log-likelihood |

## Data

`data/cook_2015_springsummer_final.csv` holds daily VWC from 39 sensor locations at the Cook Agronomy Farm (Washington State University, Pullman, WA). It covers five depths from 1 April to 30 June 2015, for 13,420 readings after removing missing values.

| Column | Description |
|---|---|
| `Location` | sensor ID |
| `Date` | measurement date |
| `Depth` | depth, scaled to [0, 1] (0.3 m to 1.5 m) |
| `VWC` | volumetric water content, scaled to [0, 1] |
| `Crop_Code` | crop, scaled to {0, 0.25, 0.5, 0.75, 1.0}; `Crop` holds the crop name |
| `x_coordinate`, `y_coordinate` | sensor position, scaled to [0, 1] |
| `Date_Normalized` | date, scaled to [0, 1] (1 April = 0, 30 June = 1) |

## Results

Mean over seeds 42 and 43 with the default settings:

| Missing sensors | MAE | RMSE | PICP | NMPIW | Paper MAE / RMSE (Table 4) |
|---|---|---|---|---|---|
| 30 % | 0.108 | 0.137 | 1.000 | 3.71 | 0.105 / 0.135 |
| 50 % | 0.125 | 0.159 | 1.000 | 3.49 | 0.113 / 0.143 |
| 70 % | 0.126 | 0.162 | 1.000 | 3.28 | 0.113 / 0.146 |

These numbers came from an NVIDIA GPU with PyTorch 2.7 and GPyTorch 1.14. On a CPU the MAE was 0.108 / 0.120 / 0.132.

**Expect small differences.** Results vary by about 0.01 to 0.015 MAE with hardware (GPU vs CPU), library versions, epochs and dropout. With only two seeds, a single split can also shift the mean: at 50 % missing the two seeds give 0.102 and 0.147. The published table was produced with an earlier version of the experiment code, so it will not match to the third decimal.

**About the uncertainty numbers.** PICP is 1.0 and NMPIW is above 1, which means the 95 % intervals are wider than the range of the test data. The model over-covers. The paper reports and discusses this in Section 4.3 and Table 5.

## Citation

```bibtex
@article{rashid2026gnngp,
  title   = {Graph-based spatio-temporal soil moisture imputation with uncertainty quantification},
  author  = {Rashid, Tamima and Majumder, Atin and Natarajan, Balasubramaniam},
  journal = {Smart Agricultural Technology},
  volume  = {15},
  pages   = {102455},
  year    = {2026},
  doi     = {10.1016/j.atech.2026.102455}
}
```
