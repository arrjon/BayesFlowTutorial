# BayesFlow Tutorial — Hands-on Exercises

A three-part introduction to **amortized Bayesian inference** with
[BayesFlow](https://bayesflow.org). Each part consists of an **exercise** notebook (with `TODO`
gaps to fill in) and a matching **solution** notebook. All notebooks are designed to run on
**Google Colab**.

> The goal: by the end you understand how to do the Bayesian workflow with BayesFlow: how
> *simulator → adapter → networks → workflow → diagnostics* fits together, and how to push it
> further with diagnostics and diffusion-model guidance.

> These exercises are built on and adapted from the official BayesFlow
> tutorials and example notebooks. For the original tutorials, full documentation, and the
> example gallery, see the BayesFlow website: <https://bayesflow.org>.

## The three parts

| # | Topic                                                                                                                                       | Level | You'll learn                                                                                                                      | Exercise | Solution |
|---|---------------------------------------------------------------------------------------------------------------------------------------------|-------|-----------------------------------------------------------------------------------------------------------------------------------|----------|----------|
| 1 | **[Linear Regression](https://colab.research.google.com/github/arrjon/BayesFlowTutorial/blob/main/01_Linear_Regression_Exercise.ipynb)**    | 🟢 Beginner | Bridge from **Pyro/NUTS** to amortized inference: same model both ways, matching posteriors, and the payoff of amortization       | [Exercise](01_Linear_Regression_Exercise.ipynb) | [Solution](01_Linear_Regression_Solution.ipynb) |
| 2 | **[SIR + Diagnostics](https://colab.research.google.com/github/arrjon/BayesFlowTutorial/blob/main/02_SIR_Diagnostics_Exercise.ipynb)**    | 🟡 Intermediate | Time-series (GRU) summaries, diagnostics, and detecting **model misspecification**        | [Exercise](02_SIR_Diagnostics_Exercise.ipynb) | [Solution](02_SIR_Diagnostics_Solution.ipynb) |
| 3 | **[Diffusion + Guidance](https://colab.research.google.com/github/arrjon/BayesFlowTutorial/blob/main/03_Diffusion_Guidance_Exercise.ipynb)** | 🔴 Advanced | Diffusion-model posteriors for a multimodal problem, and **writing your own guidance constraint** to steer sampling after training | [Exercise](03_Diffusion_Guidance_Exercise.ipynb) | [Solution](03_Diffusion_Guidance_Solution.ipynb) |

Work through them in order — each assumes the concepts from the previous one.

## Running on Google Colab

Open a notebook via its "Open in Colab" badge (top of each notebook), then run the first
setup cell. That cell installs this project and **all** its dependencies (BayesFlow, JAX,
Pyro, scikit-learn, …) straight from GitHub via
`pip install "git+https://github.com/arrjon/BayesFlowTutorial.git"` — the dependency list
lives in [`pyproject.toml`](pyproject.toml).

You may see a red pip *dependency-conflict* warning on Colab — `google-colab` pins
`pandas==2.2.2`, while BayesFlow 2.x requires `pandas>=2.3.3`. This one is unavoidable (and
harmless: the tutorial doesn't use `google-colab`'s pandas features). 

## Running locally

```bash
uv sync                      # installs everything from pyproject.toml
# or, without uv:
pip install "git+https://github.com/arrjon/BayesFlowTutorial.git"
```

BayesFlow runs on JAX, PyTorch or TensorFlow; the notebooks default to JAX.

## Further material

- BayesFlow docs & example gallery: <https://bayesflow.org>
- SBC interpretation guide: <https://hyunjimoon.github.io/SBC/articles/rank_visualizations.html>
- Diffusion models in SBI (tutorial review): <https://bayesflow-org.github.io/diffusion-experiments/>
