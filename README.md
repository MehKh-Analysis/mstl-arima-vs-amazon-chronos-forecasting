# ⚡ MSTL-ARIMA vs. Amazon Chronos Forecasting Benchmark

An end-to-end operational forecasting pipeline designed to predict high-frequency hourly electricity load **48 hours into the future**. This repository provides a rigorous benchmark evaluation comparing a classical, multi-seasonal statistical pipeline against Amazon's modern, zero-shot deep learning foundation model.

---

## 📊 Performance Summary Matrix

Evaluated on a 48-hour holdout validation set (representing complex end-of-December grid demand shocks):

| Metric | MSTL + ARIMA(2,0,0) | Amazon Chronos-Mini (Zero-Shot) | Operational Winner |
| :--- | :---: | :---: | :---: |
| **Mean Squared Error (MSE)** | 3.9900 | **1.2160** | **Amazon Chronos** 🏆 |
| **Root Mean Squared Error (RMSE)** | 1.9975 | **1.1027** | **Amazon Chronos** 🏆 |
| **Mean Absolute % Error (MAPE)** | 5.95% | **2.75%** | **Amazon Chronos** 🏆 |

### 🔍 Key Engineering Takeaways
1. **The Core Winner:** **Amazon Chronos-Mini reduced the prediction error (MAPE) by 53.7%** relative to the statistical baseline, dropping from $5.95\%$ to a highly precise $2.75\%$.
2. **Why the Statistical Pipeline Struggled:** While MSTL cleanly isolated seasonality, its *Seasonal Naive* component strictly copy-pasted past chronological waves. This linear behavior failed to adapt to non-linear holiday consumption drops occurring at the end of December.
3. **Why the Transformer Won:** Amazon Chronos treated the continuous data points as quantized numerical tokens. Its deep self-attention mechanisms successfully generalized real-world temporal anomalies without requiring local parameter re-tuning.

---

## 🛠️ Framework Architectures

### 1. Classical Pipeline: MSTL + ARIMA(2,0,0)
* **Decomposition:** Multiple Seasonal-Trend decomposition using LOESS (**MSTL**) isolates nested cycles:
  $$Y_t = T_t + S_{24} \text{ (Daily)} + S_{168} \text{ (Weekly)} + R_t \text{ (Residual)}$$
* **Seasonality:** Forward projections of $S_{24}$ and $S_{168}$ are handled via an optimized, vector-tiled (`np.tile`) chronological shift.
* **Stochastic Trend:** A non-seasonal Autoregressive **$\text{ARIMA}(2,0,0)$** model captures the remaining stationary, mean-reverting trend line ($T_t + R_t$). Parameter order was rigorously determined using **ACF/PACF lag cuts** and verified via **Ljung-Box white-noise residual testing**.

### 2. Deep Learning Pipeline: Amazon Chronos-Mini
* **Architecture:** A zero-shot time series transformer built on a T5 encoder-decoder backbone.
* **Inference Engine:** Generates $20$ distinct probabilistic simulation paths (`num_samples=20`) to map future variance. The point forecast is extracted utilizing a robust **50th-percentile median aggregation** (`np.median`) to neutralize heavy-tailed anomaly spikes.

---

## 📈 Notebook Analytical Steps
The pipeline is structured as follows:
1. **Exploratory Smoothing:** Multi-scale rolling average analysis ($24\text{h}$ vs. $168\text{h}$) to isolate hidden macro trends.
2. **Stationarity Auditing:** Augmented Dickey-Fuller (**ADF**) testing to mathematically confirm unit root boundaries ($p \approx 0$).
3. **Statistical Modeling:** MSTL structural stripping, ACF/PACF interpretation, and $\text{ARIMA}$ execution.
4. **Residual Diagnostics:** Evaluation of model health using $2\times2$ subplots (Q-Q Plots, Standardized Error, and Correlogram structures).
5. **Zero-Shot AI Deployment:** Context feeding and probabilistic decoding via `ChronosPipeline`.
6. **Unified Benchmarking:** Error calculation matrix and separate visual trend validations.

---

## 🚀 Quick Start & Installation

Clone this repository and install the dependencies:

```bash
git clone [https://github.com/YOUR_USERNAME/mstl-arima-vs-amazon-chronos-forecasting.git](https://github.com/YOUR_USERNAME/mstl-arima-vs-amazon-chronos-forecasting.git)
cd mstl-arima-vs-amazon-chronos-forecasting
pip install numpy pandas matplotlib scikit-learn statsmodels torch chronos
