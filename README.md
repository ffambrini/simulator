# Budget-constrained retention simulator (restless bandit)

Code accompanying the paper *Decision Support for Retention in Enterprise Information Systems under Delayed Data and Equity Constraints* (anonymous submission).

The simulator evaluates allocation policies for a retention program with a limited intervention capacity per period. It is instantiated for a public-sector case study (a Brazilian conditional cash transfer for upper secondary students) using state-level aggregates. All results are simulation outputs under stated assumptions; they are not estimates of program impact.

## Files

| File | Purpose |
|---|---|
| `simulador_pdm_rmab.py` | Simulator, policies, statistics and CSV output (NumPy only). |
| `simulador_pdm_rmab_graficos.py` | Same simulator plus a plotting mode that draws the figures from the CSV files (needs matplotlib). |
| `requirements.txt` | Python dependencies. |

## Requirements

Python 3.9 or later.

```
pip install -r requirements.txt
```

`simulador_pdm_rmab.py` alone needs only NumPy.

## Policies compared

`none`, `random`, `static_topk` (rank once by initial risk), `dynamic_topk` (re-rank every period by updated risk), `whittle` (Whittle index), `whittle_eq` (equity-weighted Whittle index), `ppo` (constrained PPO, learned reference, trained with a manual NumPy backpropagation) and `myopic_oracle`.

Whittle indices are computed by bisection on the Lagrange multiplier with finite-horizon value iteration, and indexability is checked numerically on a grid (reported as `indexable_frac`). The upper bound comes from a Lagrangian relaxation.

## Reproducing the reference scenario

Reference configuration: N = 20,000 students per cohort, T = 9 months, mean effect 0.3, risk drift 0.05, effect persistence 0.10, capacity k/N = 5%, mean latent dropout 8%, scenario S2, 30 paired seeds (base 2027).

```
python simulador_pdm_rmab.py --mode run --eps-profile homog --eps 0.3 --k-frac 0.05 --out out_homog
```

Without the PPO (fast, about 10 seconds):

```
python simulador_pdm_rmab.py --mode run --eps-profile homog --no-ppo --out out_homog
```

Other effect profiles: `--eps-profile hump` (most effective at intermediate risk) and `--eps-profile decl` (declining with risk).

Equity frontier (Pareto curve between total and worst-group retention):

```
python simulador_pdm_rmab.py --mode run --eps-profile homog --kappa-grid 0.5,1,2,4 --out out_pareto
```

Grid of configurations (profile x drift x persistence x capacity x effect = 243 configurations):

```
python simulador_pdm_rmab.py --mode sweep --no-ppo --sweep-eps 0.2,0.3,0.4 --jobs 2 --out out_sweep
```

Latent dropout versus administrative lag scenarios are selected with `--scenario S0|S1|S2|S3`.

## Figures

The plotting version writes PNG figures at the end of `run` and `sweep` (disable with `--no-plots`), or from existing CSV files:

```
python simulador_pdm_rmab_graficos.py --mode plots --out out_homog
```

## Output files (written to `--out`)

| File | Content |
|---|---|
| `resultados_por_seed.csv` | Metrics per policy and seed. |
| `retencao_por_uf.csv` | Retention per state, policy and seed. |
| `resumo.csv` | Mean and 95% interval per policy. |
| `diferencas_pareadas.csv` | Paired differences per seed: normal 95% interval, percentile bootstrap, equivalence test (TOST, margin of 2 students). |
| `sweep_mapa.csv`, `sweep_pareado.csv` | Results of the configuration grid. |

## Assumptions and limitations

- State risk rates (`UF_RATE`) and state weights (`UF_POP`) are approximate values in the code and should be replaced by the actual cohort counts for any use beyond reproducing the paper's qualitative findings.
- The mean dropout level, the risk dynamics, the intervention effect and its profile by risk level are modeling assumptions that cannot be identified from the open data; they are handled by sensitivity analysis.
- Seeds are fixed (`--seed`, default 2027), so runs are reproducible on the same NumPy version.

## License

To be added after the review.
