# 22f3000284@ds.study.iitm.ac.in
#
# Notebook goals:
# - Demonstrate the relationship between variables in a synthetic dataset.
# - Include interactive slider widgets to control data generation.
# - Show dynamic markdown that updates based on widget state.
# - Be self-documenting with comments describing the data flow.
#
# Data flow overview (cell → cell dependencies):
# [Cell 1] Imports & RNG  ─┐
#                          ├─> [Cell 3] Widgets (sliders) ─┐
# [Cell 2] Widgets UI  ────┘                              ├─> [Cell 4] Generate dataset df(x,y)
#                                                         ├─> [Cell 6] Dynamic markdown report
# [Cell 1] Imports & RNG  ────────────────────────────────┘
# [Cell 5] Summary(df)  depends on [Cell 4]
# [Cell 7] OLS fit(beta_hat) depends on [Cell 4]
# [Cell 8] Plot(df, beta_hat) depends on [Cell 4] and [Cell 7]
#
# How to run:
#   1) pip install marimo
#   2) marimo run analysis.py
#
# Note: Marimo runs each cell reactively. Changing a slider updates cells that depend on it.

import marimo as mo

app = mo.App()

@app.cell
def __():
    # [Cell 1] Core imports and a reproducible random number generator.
    # Downstream dependencies: Cells 4, 8.
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt

    # Reproducible RNG for noise generation.
    rng = np.random.default_rng(42)
    return np, pd, plt, rng


@app.cell
def __():
    # [Cell 2] Define interactive widgets (sliders) controlling the data-generating process.
    # Downstream dependencies: Cells 4, 6.
    slope = mo.ui.slider(min=0.1, max=5.0, step=0.1, value=2.0, label="True slope (β)")
    noise = mo.ui.slider(min=0.0, max=5.0, step=0.1, value=1.0, label="Noise standard deviation (σ)")
    n = mo.ui.slider(min=20, max=500, step=10, value=120, label="Sample size (n)")

    # Display the controls stacked vertically.
    mo.vstack([slope, noise, n])
    return slope, noise, n


@app.cell
def __(np, rng, slope, noise, n, pd):
    # [Cell 4] Generate synthetic dataset df based on widget state.
    # Upstream: Cells 1 (np, rng), 2 (slope, noise, n).
    # Downstream: Cells 5, 7, 8.

    x = np.linspace(0, 10, int(n.value))
    eps = rng.normal(0.0, float(noise.value), size=x.shape[0])
    y = float(slope.value) * x + eps

    df = pd.DataFrame({"x": x, "y": y})
    return df


@app.cell
def __(df):
    # [Cell 5] Summarize the generated dataset.
    # Upstream: Cell 4 (df). Downstream: none (display only).
    desc = df.describe()
    desc


@app.cell
def __(df, np):
    # [Cell 7] Fit an OLS line y = b0 + b1*x using least squares.
    # Upstream: Cell 4 (df). Downstream: Cell 8 (plot), Cell 6 (report).
    X = np.vstack([np.ones(len(df)), df["x"].values]).T
    y_vec = df["y"].values
    beta_hat = np.linalg.lstsq(X, y_vec, rcond=None)[0]  # [b0, b1]
    return beta_hat


@app.cell
def __(df, plt, beta_hat, np):
    # [Cell 8] Plot scatter and fitted regression line.
    # Upstream: Cell 4 (df), Cell 7 (beta_hat). Downstream: none (display only).
    fig, ax = plt.subplots()
    ax.scatter(df["x"], df["y"], alpha=0.7)
    xline = np.array([df["x"].min(), df["x"].max()])
    yline = beta_hat[0] + beta_hat[1] * xline
    ax.plot(xline, yline, linewidth=2)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_title("Synthetic data & OLS fit")
    fig


@app.cell
def __(slope, noise, n, beta_hat, mo):
    # [Cell 6] Dynamic markdown that updates when widgets change.
    # Upstream: Cell 2 (slope, noise, n), Cell 7 (beta_hat). Downstream: none (display only).
    mo.md(f"""
### Interactive report

- **True slope (β)** from slider: **{float(slope.value):.2f}**
- **Noise σ** from slider: **{float(noise.value):.2f}**
- **Sample size (n)** from slider: **{int(n.value)}**

**Estimated intercept (b₀)**: {beta_hat[0]:.3f}  
**Estimated slope (b₁)**: {beta_hat[1]:.3f}

**Interpretation:** Increasing **σ** (noise) widens the scatter around the line, which can make the
estimated slope less stable. Increasing **n** (sample size) typically tightens confidence around
the true β, so the estimated b₁ tends to move closer to the slider's **β**.
""")


@app.cell
def __():
    # [Cell 3] Self-documenting "how to run" cell.
    import marimo as mo
    mo.md("""
### How to run this notebook

1. Install dependencies:
   ```bash
   pip install marimo
   ```
2. Launch the reactive notebook:
   ```bash
   marimo run analysis.py
   ```
3. Adjust the sliders to see how the dataset and the regression update live.
""")


if __name__ == "__main__":
    app.run()
