# BayesFlow Tutorial — Hands-on Exercises

A three-part, increasing-difficulty introduction to **amortized Bayesian inference** with
[BayesFlow](https://bayesflow.org). Each part ships as an **exercise** notebook (with `TODO`
gaps to fill in) and a matching **solution** notebook. All notebooks are designed to run on
**Google Colab** — the first cell installs everything you need.

> The goal: by the end you understand what BayesFlow is, how the pipeline
> *simulator → adapter → networks → workflow → diagnostics* fits together, and how to push it
> further with diagnostics and diffusion-model guidance.

> ℹ️ **Attribution.** These exercises are **built on and adapted from the official BayesFlow
> tutorials and example notebooks**. For the original tutorials, full documentation, and the
> example gallery, see the BayesFlow website: <https://bayesflow.org>.

## The three parts

| # | Topic | Level | You'll learn | Exercise | Solution |
|---|-------|-------|--------------|----------|----------|
| 1 | **Linear Regression** | 🟢 Beginner | Bridge from **Pyro/NUTS** (yesterday's tool) to amortized inference: same model both ways, matching posteriors, and the payoff of amortization | [Exercise](01_Linear_Regression_Exercise.ipynb) | [Solution](01_Linear_Regression_Solution.ipynb) |
| 2 | **SIR + Diagnostics** | 🟡 Intermediate | Time-series (GRU) summaries, the full diagnostic suite, non-identifiability, and detecting **model misspecification** with a second "weekend-delay" simulator | [Exercise](02_SIR_Diagnostics_Exercise.ipynb) | [Solution](02_SIR_Diagnostics_Solution.ipynb) |
| 3 | **Diffusion + Guidance** | 🔴 Advanced | Diffusion-model posteriors for a multimodal problem, and **writing your own guidance constraint** to steer sampling after training | [Exercise](03_Diffusion_Guidance_Exercise.ipynb) | [Solution](03_Diffusion_Guidance_Solution.ipynb) |

Work through them in order — each assumes the concepts from the previous one.

## Running on Google Colab

Open a notebook via its "Open in Colab" badge (top of each notebook), then run the first
setup cell. 

> 📎 **TODO (maintainer):** the Colab badges and the `misc/` clone command currently point at
> a placeholder repository path (`TODO-your-org/BayesFlowTutorial`). Search-and-replace it with
> your real GitHub `org/repo` once this is pushed so the badges and the notebook-3 clone work.

## Running locally

```bash
uv sync          # or: pip install "bayesflow>=2.0.12" jax
# notebook 1 also needs: pyro-ppl (+ torch)
# notebook 3 also needs: scikit-learn scipy
```

BayesFlow runs on JAX, PyTorch or TensorFlow; the notebooks default to JAX.

## Further material

- BayesFlow docs & example gallery: <https://bayesflow.org>
- SBC interpretation guide: <https://hyunjimoon.github.io/SBC/articles/rank_visualizations.html>
- Diffusion models in SBI (tutorial review): <https://bayesflow-org.github.io/diffusion-experiments/>

> 📎 **TODO (course):** add links to the accompanying lecture slides / recordings here.
