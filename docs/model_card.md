# Model card

## Intended use

This model stack is a research tool for studying whether filing-date-clean institutional ownership-network signals forecast the cross-section of stock returns. It is not investment advice and is not a production trading system.

## Training and evaluation

- Signal panel: Step 005 full filing-date-clean ownership-network panel.
- Baselines: deciles/rank ICs in Step 006 and Fama-MacBeth regressions in Step 007.
- OOS ML: expanding-window Step 008 models with embargoes.
- Portfolio evaluation: Step 009 transaction-cost and factor-attribution layer.
- Robustness: Step 010 regime, stratum, network-incremental, and placebo diagnostics.

## Headline model

- model: `ridge_all`
- target: `fwd_ret_1m`
- annualized net return at 25 bps: **7.88%**
- Sharpe approximation: **0.795**
- Newey-West t-stat: **2.65**

## Limitations

- 13F data are delayed, quarterly, amended, and sometimes affected by confidential treatment.
- Returns are backtest diagnostics and do not include implementation frictions beyond the specified turnover-cost model.
- Vendor-derived panels are local-only and not redistributed.
