# Research Report — Finance Data Analysis Lab

**Descriptive quantitative analysis of daily equity market data: transforming raw prices into interpretable risk and return information**

*Author: Aydin Monavvari — educational research project (not investment advice).*
*Run date: 2026-10-08. All numbers in this report are actual outputs of the committed pipeline.*

---

## Abstract

This project develops a reproducible Python pipeline that converts raw daily adjusted price data for seven U.S. large-capitalization equities and the SPDR S&P 500 ETF (SPY) benchmark into a structured set of risk and return statistics: annualized returns and volatilities, the Sharpe, Sortino and Calmar ratios, maximum drawdown, market beta and alpha, and formal statistical diagnostics (Jarque–Bera normality tests, Augmented Dickey–Fuller stationarity tests, and Ljung–Box autocorrelation tests). Over the sample 2018-01-02 to 2026-10-07 (2,203 trading days), the analysis documents several well-known stylized facts of equity markets in the actual data: daily log returns are strongly non-normal (excess kurtosis between 4.6 and 13.7; Jarque–Bera rejects normality at the 5% level for all eight instruments), log prices behave like unit-root (non-stationary) processes while log returns are stationary, and significant short-horizon autocorrelation is present in most return series. The purpose of the project is methodological: to demonstrate that careful, validated quantitative analysis can transform unstructured price data into interpretable investment and risk information, and to make every step of that transformation transparent and reproducible. The project is descriptive; it makes no predictive claims and provides no investment recommendations.

## Introduction

Raw financial market data — a column of adjusted closing prices — carries no meaning by itself. Investment and risk decisions require *derived* quantities: how much value was created (return), how uncertain that value is (volatility, drawdown), how an asset co-moves with the market (beta, correlation), and whether the statistical properties of the series justify the use of standard models (normality, stationarity, autocorrelation). Building these quantities correctly is harder than it appears: annualization conventions, risk-free-rate assumptions, log versus simple returns, and missing-data policies all involve choices that materially change the resulting numbers and must be documented rather than hidden.

This project implements that transformation as a small, tested, end-to-end pipeline. It is the foundational repository in a research portfolio that progressively builds toward econometrics, machine learning, deep learning, NLP and LLM-based systems for financial and economic research: before any model can be trained or any strategy evaluated, the basic language of quantitative finance — returns, risk, co-movement, and the statistical behavior of price series — must be handled correctly and reproducibly.

## Research Question

> How can quantitative analysis transform raw financial market data into interpretable investment and risk information?

The operational form of this question is: given only daily adjusted prices, which set of well-defined statistics and tests yields the most informative, defensible description of each asset's risk–return profile, and what do those statistics actually reveal about the sample?

## Related Work

The metrics implemented here rest on a long literature:

- **Markowitz, H. (1952).** "Portfolio Selection." *Journal of Finance*, 7(1), 77–91. Introduces the mean–variance framework in which expected return, variance and covariance are the primitive objects of investment analysis; the covariance and correlation matrices computed here are the inputs that framework consumes.
- **Sharpe, W. F. (1966).** "Mutual Fund Performance." *Journal of Business*, 39(1), 119–138. Introduces the reward-to-variability ratio (Sharpe ratio) used here as the primary risk-adjusted performance measure.
- **Fama, E. F. (1970).** "Efficient Capital Markets: A Review of Theory and Empirical Work." *Journal of Finance*, 25(2), 383–417. Frames the question of how much information prices already contain; motivates why a purely descriptive project makes no predictive claims.
- **Jarque, C. M., & Bera, A. K. (1980).** "Efficient tests for normality, homoscedasticity and serial independence of regression residuals." *Economics Letters*, 6(3), 255–259. The normality test applied to daily returns.
- **Dickey, D. A., & Fuller, W. A. (1979).** "Distribution of the Estimators for Autoregressive Time Series with a Unit Root." *Journal of the American Statistical Association*, 74(2), 427–431. Basis of the ADF test used to distinguish stationary returns from non-stationary price levels.
- **Ljung, G. M., & Box, G. E. P. (1978).** "On a Measure of a Lack of Fit in Time Series Models." *Biometrika*, 65(2), 297–303. Basis of the portmanteau autocorrelation test.

## Data

| Property | Value (actual run) |
| --- | --- |
| Instruments | AAPL, MSFT, JNJ, JPM, XOM, PG, AMZN + benchmark SPY |
| Source | Yahoo Finance daily bars, retrieved via the `yfinance` Python package |
| Field | Adjusted close (split- and dividend-adjusted) |
| Window | 2018-01-02 → 2026-10-07 |
| Observations | 2,203 trading days per instrument |
| Missing values | 0 before cleaning; forward-fill limit of 5 days defined but never triggered |
| Validation | Monotonic business-day index enforced; non-positive prices rejected; 0 rows dropped |

Provenance and licensing notes: Yahoo Finance data is supplied for personal, non-commercial use; the project caches raw CSVs locally under `data/raw/` (git-ignored) so the analysis is fully offline-reproducible from the cache, and documents the acquisition script for re-download. Fundamentals (balance-sheet ratios) were deliberately excluded because free programmatic sources do not provide sufficiently reliable point-in-time coverage; the project scope is therefore price-based analytics.

## Methodology

**Returns.** Simple returns $R_t = P_t/P_{t-1} - 1$ and log returns $r_t = \ln(P_t/P_{t-1})$ are both computed; log returns are used for statistical diagnostics because they are additive over time.

**Annualization.** Mean returns and volatilities are annualized with a 252-trading-day convention: $\mu_{ann} = 252\,\bar r$, $\sigma_{ann} = \sqrt{252}\,\sigma_r$. The convention is a configuration parameter, not a hidden constant.

**Risk-adjusted performance.** With a configurable risk-free assumption ($r_f = 0$ in the default run, stated rather than hidden):

- Sharpe ratio: $(\mu_{ann} - r_f)/\sigma_{ann}$
- Sortino ratio: $(\mu_{ann} - r_f)/\sigma_{downside}$, using downside deviation only
- Calmar ratio: annualized return / maximum drawdown

**Drawdown.** The drawdown series is $DD_t = P_t/\max_{s \le t} P_s - 1$; maximum drawdown is its minimum.

**Market co-movement.** Beta is the OLS slope of asset daily returns on SPY daily returns; annualized alpha is the intercept scaled by 252. Full correlation and covariance matrices of daily returns are computed.

**Statistical diagnostics.** Jarque–Bera tests of normality of daily log returns; Augmented Dickey–Fuller tests applied separately to log prices (expected: unit root not rejected) and log returns (expected: unit root rejected); Ljung–Box test at lag 20 for residual autocorrelation.

**Engineering.** All routines are typed, documented, unit-tested against hand-computed values (43 offline tests), and orchestrated by a deterministic pipeline writing machine-readable outputs (`reports/metrics.json`) and human-readable summaries (`reports/summary.md`).

## Experimental Design

The design is descriptive, not predictive: there is no train/test split and no model fitting beyond OLS beta, because there is nothing being forecast. The analytical choices that would be confounded with prediction (look-ahead bias, data snooping) do not arise; the relevant validity threats here are *measurement* threats, addressed as follows:

1. **Adjusted prices** are used throughout so that total-return-relevant corporate actions do not distort return calculations.
2. **A single, documented annualization convention** is applied to every instrument.
3. **The risk-free assumption is explicit** ($r_f = 0$) and configurable; with $r_f > 0$ all Sharpe ratios would shift down uniformly, which is why the *ranking* is robust while absolute levels are assumption-dependent.
4. **Validation gates** (index monotonicity, non-positive prices, missing-value policy) run before any statistic is computed, and their outcome is recorded in the report file.
5. **Every figure is generated by code from the same cached data** used for the tables — no hand-edited artifacts.

## Results

Actual pipeline output (2026-10-08; $r_f = 0$; sample 2018-01-02 → 2026-10-07, 2,203 days), sorted by Sharpe ratio:

| Ticker | Ann. return | Ann. vol | Sharpe | Sortino | Calmar | Max drawdown | Beta (vs SPY) | Alpha (ann.) |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| AAPL | 27.52% | 30.43% | 0.95 | 1.40 | 0.71 | −38.52% | 1.20 | 10.34% |
| MSFT | 24.41% | 29.08% | 0.90 | 1.33 | 0.66 | −37.15% | 1.17 | 7.90% |
| SPY | 14.65% | 18.98% | 0.82 | 1.15 | 0.43 | −33.72% | 1.00 | 0.00% |
| JPM | 16.76% | 28.52% | 0.69 | 1.01 | 0.38 | −43.63% | 1.05 | 3.26% |
| AMZN | 18.39% | 34.43% | 0.66 | 0.98 | 0.33 | −56.15% | 1.20 | 4.24% |
| JNJ | 10.32% | 19.63% | 0.60 | 0.85 | 0.38 | −27.37% | 0.43 | 5.05% |
| XOM | 12.68% | 29.97% | 0.55 | 0.79 | 0.21 | −61.01% | 0.75 | 4.88% |
| PG | 8.62% | 20.03% | 0.51 | 0.74 | 0.36 | −23.77% | 0.47 | 2.95% |

Statistical diagnostics (all instruments, daily log returns):

| Diagnostic | Actual finding |
| --- | --- |
| Jarque–Bera (normality) | Rejected at 5% for all 8 series; excess kurtosis ranges from 4.63 (AMZN) to 13.67 (SPY) |
| ADF on log prices | Unit root **not** rejected for any instrument (p-values 0.37–0.96) — price levels are non-stationary |
| ADF on log returns | Unit root rejected for all 8 series (p < 0.001) — returns are stationary |
| Ljung–Box (lag 20) | Autocorrelation rejected for 7 of 8 series; only AMZN shows no significant linear autocorrelation (p = 0.129) |

The six committed figures visualize these results: indexed price paths, return histograms against fitted normals (visibly leptokurtic), cumulative return comparison vs SPY, drawdown curves, the return correlation heatmap, and rolling 63-day volatility.

## Discussion

Three findings deserve emphasis because they generalize beyond this sample:

1. **The two technology names dominate on risk-adjusted performance** (Sharpe 0.95 and 0.90 vs 0.82 for the market), but they also carry the highest market exposure (β ≈ 1.2). Their alpha over SPY (10.3% and 7.9% annualized) is a *descriptive* statement about this particular eight-year window, in which large-cap technology outperformed; it is not evidence of persistent skill or mispricing.
2. **Defensive names behave as expected structurally**: JNJ and PG show the lowest betas (0.43, 0.47), the shallowest drawdowns (−27.4%, −23.8%) and the lowest volatilities, confirming that sector characteristics, not only firm size, drive co-movement.
3. **The statistical diagnostics paint the standard picture of financial returns** that motivates the entire downstream portfolio: non-normal (fat-tailed) returns invalidate Gaussian risk assumptions (a VaR model calibrated to a normal distribution would systematically understate tail risk here); stationarity of returns but not of prices justifies modeling returns rather than levels in the later forecasting projects; and significant Ljung–Box statistics indicate small but detectable linear predictability — the phenomenon that the market-prediction study in this portfolio tests under strict out-of-sample discipline.

The drawdown results also illustrate the difference between volatility and loss experience: XOM's maximum drawdown (−61.0%) is nearly triple PG's (−23.8%) while their annualized volatilities differ by only 10 percentage points — drawdown captures path dependence that a single volatility number cannot.

## Limitations

- **Data quality.** Yahoo Finance adjusted prices are delayed, potentially revised, and not an official exchange feed. The project's conclusions are about the cached snapshot analyzed.
- **Survivorship/selection bias.** The seven names are today's large caps that were also large caps in 2018; delisted or distressed firms are absent, so cross-sectional summaries (e.g., average Sharpe) are upward-biased relative to the tradable universe of 2018.
- **No point-in-time fundamentals.** Financial ratios were excluded by design; the project therefore says nothing about valuation.
- **Risk-free rate simplification.** $r_f = 0$ overstates Sharpe ratios in high-rate periods; the constant is configurable and disclosed.
- **Single convention risk.** 252-day annualization, logarithmic returns for diagnostics and OLS beta are conventions; other defensible choices would shift levels (not orderings) modestly.
- **No microstructure.** Daily closes ignore intraday risk, liquidity, and transaction costs; nothing here estimates tradable performance.

## Conclusion

A disciplined, fully tested pipeline converts raw adjusted prices into a coherent risk–return profile for eight instruments and recovers the canonical stylized facts of equity markets in the actual data: fat-tailed, stationary returns; non-stationary price levels; small but significant serial correlation; and strong dispersion in drawdown experience across sectors. The repository demonstrates that the *foundational layer* of quantitative finance — measurement — can be engineered with the same rigor as a model: validated inputs, explicit conventions, unit-tested statistics, and reproducible outputs.

## Future Research

- Extend to a multi-asset universe with point-in-time index membership to remove selection bias.
- Add realized-variance estimators from intraday data and compare against daily-close volatility.
- Feed the computed covariance matrix into the portfolio-optimization project (mean–variance and risk-parity) as a downstream consumer.
- Replace OLS beta with rolling and Bayesian beta to expose estimation instability.
- Add a formal multiple-testing framework when the universe grows (Bonferroni/Holm corrections across instruments).

## References

- Dickey, D. A., & Fuller, W. A. (1979). Distribution of the estimators for autoregressive time series with a unit root. *Journal of the American Statistical Association*, 74(2), 427–431.
- Fama, E. F. (1970). Efficient capital markets: A review of theory and empirical work. *Journal of Finance*, 25(2), 383–417.
- Jarque, C. M., & Bera, A. K. (1980). Efficient tests for normality, homoscedasticity and serial independence of regression residuals. *Economics Letters*, 6(3), 255–259.
- Ljung, G. M., & Box, G. E. P. (1978). On a measure of a lack of fit in time series models. *Biometrika*, 65(2), 297–303.
- Markowitz, H. (1952). Portfolio selection. *Journal of Finance*, 7(1), 77–91.
- Sharpe, W. F. (1966). Mutual fund performance. *Journal of Business*, 39(1), 119–138.
