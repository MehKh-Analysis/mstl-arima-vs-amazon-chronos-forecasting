# ⚡ MSTL-ARIMA vs. Amazon Chronos Forecasting Benchmark

An end-to-end forecasting pipeline that predicts hourly electricity demand **48 hours into the future**, comparing a classical multi-seasonal statistical pipeline against Amazon's zero-shot deep learning foundation model.

---

## 📂 Data

Hourly electricity demand from a regional power grid. The final 48 hours of the series (late December, including holiday-period demand shifts) are held out for evaluation.

---

## 📊 Performance Summary

Evaluated on the 48-hour holdout:

| Metric                             | MSTL + ARIMA(2,0,0) | Amazon Chronos-Mini (Zero-Shot) | Winner   |
| ---------------------------------- | ------------------- | ------------------------------- | -------- |
| **Mean Squared Error (MSE)**       | 3.9900              | **1.2160**                      | **Chronos** 🏆 |
| **Root Mean Squared Error (RMSE)** | 1.9975              | **1.1027**                      | **Chronos** 🏆 |
| **Mean Absolute % Error (MAPE)**   | 5.95%               | **2.75%**                       | **Chronos** 🏆 |

### 🔍 Key Takeaways

1. **Chronos-Mini reduced MAPE by 53.7%** relative to the statistical baseline, from 5.95% to 2.75%.
2. **Why the statistical pipeline likely struggled:** MSTL cleanly isolated the daily and weekly seasonality, but projecting those patterns forward repeats past cycles. That approach can't adapt to the unusual holiday-period demand at the end of December.
3. **Why Chronos may have done better:** Chronos is pretrained on a large, diverse collection of time series, which may help it handle irregular patterns without local tuning. Confirming this would require testing across more periods (see Limitations).

---

## 🛠️ Model Pipelines

### 1. Classical Pipeline: MSTL + ARIMA(2,0,0)

- **Decomposition:** Multiple Seasonal-Trend decomposition using LOESS (**MSTL**) separates the nested cycles:
$$Y_t = T_t + S_{24} \text{ (Daily)} + S_{168} \text{ (Weekly)} + R_t \text{ (Residual)}$$
- **Seasonality:** The daily ($S_{24}$) and weekly ($S_{168}$) components are projected forward by repeating their most recent cycles (`np.tile`).
- **Trend + residual:** A non-seasonal **$\text{ARIMA}(2,0,0)$** model captures the remaining trend and residual ($T_t + R_t$). The order was chosen from **ACF/PACF** plots and checked with a **Ljung-Box** test on the residuals.

### 2. Deep Learning Pipeline: Amazon Chronos-Mini

- **Architecture:** A zero-shot time series transformer built on a T5 encoder-decoder.
- **Inference:** Generates 20 probabilistic sample paths (`num_samples=20`); the point forecast is their **median**, which is robust to outlier paths.

---

## 📈 Notebook Steps

1. **Exploratory smoothing:** rolling averages (24h vs. 168h) to reveal longer-term trends.
2. **Stationarity check:** Augmented Dickey-Fuller (**ADF**) test.
3. **Statistical modeling:** MSTL decomposition, ACF/PACF analysis, and ARIMA fitting.
4. **Residual diagnostics:** Q-Q plot, standardized residuals, and correlogram.
5. **Zero-shot forecasting:** Chronos inference with `ChronosPipeline`.
6. **Benchmarking:** error metrics and forecast-vs-actual plots for both models.

---

## ⚠️ Limitations

- **Single evaluation window.** Results come from one 48-hour holdout. A rolling-origin backtest over many windows would show whether Chronos's advantage holds consistently.
- **Single dataset.** Results may differ on other grids or regions.
- **Simple seasonal projection.** The classical baseline repeats past seasonal cycles and has no holiday or calendar features, which likely disadvantages it in late December.
- **Point forecasts only.** Chronos produces probabilistic forecasts, but only the median is evaluated here.

---

## 🚀 Quick Start

```bash
git clone https://github.com/MehKh-Analysis/mstl-arima-vs-amazon-chronos-forecasting.git
cd mstl-arima-vs-amazon-chronos-forecasting
pip install -r requirements.txt
```

Then open `notebook.ipynb` and run all cells.
