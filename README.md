# MLOps long-term stock prediction: 6-month stock excess-return forecasting

Investors who hold stocks for months need to know not only the expected return of a stock but how uncertain it is.
This project builds an ML system that predicts the 10th, 25th, 50th, 75th and 90th percentiles of a stock's 6-month return relative to the S&P 500, from price history and SEC filings.
The predictions are produced for about 500 large US stocks and served to a portfolio analyst, together with the top features behind each forecast.
Because the true return is only known 6 months after each prediction, the system monitors feature drift, interval width and, as labels mature, quantile coverage.
The goal of the course project is not a profitable model but a reproducible, integrated and observed system, built over five labs (design, pipelines, LLM component, serving and monitoring, agent).

Design document: [docs/design.md](docs/design.md)

## Quick start

```
make setup   # install the pinned environment (uv)
make lint    # run ruff
```

## Maturity scorecard

| Property | Evidence a reviewer checks | Added in |
| --- | --- | --- |
| Repository reproducible from a clean clone | CI is green; `make setup` and `make lint` succeed on a fresh clone | Lab 1 |
| Design document exists and is current | `docs/design.md` follows the template, every section filled or marked as an assumption | Lab 1 |
