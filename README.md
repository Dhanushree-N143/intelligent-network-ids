# GWO-Based Network Intrusion Detection System

**NSL-KDD | Binary Grey Wolf Optimization | Random Forest**

## 1. Project overview
Academic project: *Intelligent Network Intrusion Detection Using Metaheuristic Optimization*.

This repository contains the **existing** IDS implementation (a paper-inspired baseline, not an exact reproduction of any
single paper) and a Streamlit dashboard that demonstrates it:

```
NSL-KDD -> Preprocessing -> Binary GWO -> Feature Selection -> Random Forest -> Intrusion Classification
```

The fitness function is `0.99 * (1 - validation accuracy) + 0.01 * (selected features / total features)`, evaluated on a
stratified 80/20 holdout of the working training set. Final models are evaluated on `KDDTest+`.

> This is a demonstration of an **existing** system. It is not the proposed/novel IDS and makes no novelty claim.

## 2. Folder structure
```
GWO_IDS/
├── app.py                  Streamlit dashboard (UI only, no algorithm logic)
├── gwo_ids.py              Existing IDS engine + CLI (GWO + Random Forest, baseline mode)
├── requirements.txt
├── README.md
├── .streamlit/config.toml  Dashboard theme
├── data/                   KDDTrain+.arff, KDDTest+.arff
├── results/                Files written by the IDS (read by the dashboard)
│   ├── baseline_results.csv, baseline_selected_features.csv
│   ├── gwo_results.csv, gwo_selected_features.csv, gwo_convergence.csv
│   ├── gwo_predictions.csv, gwo_feature_importance.csv, gwo_run_metadata.json     (written by new runs)
│   ├── baseline_predictions.csv, baseline_feature_importance.csv, baseline_run_metadata.json
│   └── archive_original_zip_results/   untouched copy of the results that came with the original ZIP
└── experiments/            Team-recorded development experiments (never overwritten by runs)
    ├── seed_experiments.csv, iteration_experiments.csv, reference_runs.csv, README.md
```

## 3. Installation
Python 3.9+ is recommended.
```
python -m pip install -r requirements.txt
```

## 4. Dataset placement
Place the NSL-KDD ARFF files in `data/`:
```
data/KDDTrain+.arff
data/KDDTest+.arff
```

## 5. Run the CLI IDS (GWO + Random Forest)
```
python gwo_ids.py --train-arff data/KDDTrain+.arff --test-arff data/KDDTest+.arff --sample 5000 --pop 5 --iters 10 --trees 50 --seed 42
```
Larger run:
```
python gwo_ids.py --train-arff data/KDDTrain+.arff --test-arff data/KDDTest+.arff --sample 10000 --pop 8 --iters 15 --trees 50
```
Output is written to `results/`. Console output is the same as before.

## 6. Run the all-feature baseline (no GWO)
```
python gwo_ids.py --train-arff data/KDDTrain+.arff --test-arff data/KDDTest+.arff --baseline
```

## 7. Launch the dashboard
```
streamlit run app.py
```
It opens at http://localhost:8501.

## 8. Example commands
```
python gwo_ids.py --train-arff data/KDDTrain+.arff --test-arff data/KDDTest+.arff --seed 1 --iters 10
python gwo_ids.py --train-arff data/KDDTrain+.arff --test-arff data/KDDTest+.arff --baseline --sample 5000 --seed 42
streamlit run app.py
```

## 9. Dashboard sections
| Page | What it shows | Data source |
|---|---|---|
| 1 System Overview | Architecture, metric cards, implementation details (expander) | `results/gwo_results.csv`, run metadata, dataset files |
| 2 Feature Selection | 41 -> selected -> reduction, binary selection map, feature table | `results/gwo_selected_features.csv` |
| 3 Performance Comparison | All Features + RF vs GWO + RF: grouped bars, table, runtime, feature count | `results/baseline_results.csv`, `results/gwo_results.csv` |
| 4 GWO Convergence | Best-fitness line chart, best fitness / iteration, iteration-budget log | `results/gwo_convergence.csv`, `experiments/iteration_experiments.csv` |
| 5 Experimental Stability | Seed 1-3 development experiments | `experiments/seed_experiments.csv` |
| 6 Confusion Matrix & Importance | Confusion matrix from saved predictions; Random Forest feature importance | `results/gwo_predictions.csv`, `results/gwo_feature_importance.csv` |
| 7 Run IDS | Run GWO IDS / Run All-Feature Baseline with real progress messages | calls `gwo_ids.run_gwo` / `gwo_ids.run_baseline` |

Nothing in the dashboard is hard-coded: if a file is missing, the page says which file is missing and how to create it.
Running from the dashboard first copies the current result files into `results/backup_<timestamp>/` (can be unticked).

**Note:** the confusion matrix and feature-importance charts need `gwo_predictions.csv` / `gwo_feature_importance.csv`,
which the original implementation did not save. They appear after one run of the IDS (dashboard button or CLI).

**Recommended 3-5 minute review flow:** 1 Overview -> 2 Feature Selection -> 3 Performance -> 4 Convergence -> 6 Confusion Matrix -> 7 Run IDS.

## 10. Important limitations
* Demonstration of an existing implementation; no novelty is claimed.
* Results are single-run development measurements on a 5,000-row training sample by default; they are not final benchmarks
  and are not statistical estimates. Different seeds select different feature subsets and give different metrics (page 5).
* Results can differ across machines and library versions (scikit-learn / pandas). Library versions are saved in
  `results/*_run_metadata.json` for every new run.
* The `experiments/` CSVs are the team's recorded development log, not outputs of the dashboard.
* In the recorded seed-1 iteration experiment, the best recorded fitness was reached in the first iteration and remained
  unchanged in later iterations for that configuration. This is an observation, not a diagnosed convergence problem.
* Scalability and adaptation to dynamic network environments have **not** been evaluated.
* Runtime figures include data loading and preprocessing; they are not a computational-complexity analysis.
* Random Forest feature importance (page 6) describes the final classifier, not the importance of features in the GWO search.

## Changes relative to the original implementation
`gwo_ids.py`: the algorithm (BinaryGWO, fitness, preprocessing, Random Forest settings) is unchanged. `main()` was split into
`run_gwo()` / `run_baseline()` so the dashboard can call the same code; optional progress hooks were added; and extra files
(predictions, importances, run metadata) are saved alongside the original CSVs.
