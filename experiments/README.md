# Development experiments (team-recorded)

These CSV files record the **development experiments conducted by the team using different random
seeds / iteration budgets** on the existing GWO + Random Forest implementation.

* They were supplied by the project team as a written experiment log; they are **not** produced
  by the dashboard and are **not** final benchmark results.
* They live here (not in `results/`) so that running the IDS from the dashboard or CLI never
  overwrites them.
* Values are stored as fractions (0.8035 = 80.35 %) to match the format of `results/*.csv`.
* Results can differ slightly between machines/library versions; re-running the IDS on a
  different environment may not reproduce these numbers exactly.
