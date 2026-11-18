# CorrConvNet-1D – code and results
- code/: pip install torch scikit-learn lightgbm lime scipy pandas matplotlib
  Data: 11_Tumors CSV from github.com/simonorozcoarias/ML_DL_microArrays (data11tumors2.csv); Leukemia_GSE9476.csv from CuMiDa. Edit paths in data.py.
  Run: each line of jobs.txt is `python run.py <args>` (≈1 h per 11_Tumors repeat on 1 CPU core); then `python analyze.py 11tumors`, `python analyze.py leukemia`, `python figs.py`, `python make_tables.py`.
- results/: per-fold, per-sample probabilities for all models (pickles) and summary.json with all statistics.
