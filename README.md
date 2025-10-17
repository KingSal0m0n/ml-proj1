# Project 1 — ML course (ex03)

This repo contains my implementations for project 1.

## Files
- `implementations.py` — required functions (GD/SGD, least squares, ridge, logistic & regularized logistic)
- `costs.py`, `gradient_descent.py`, `stochastic_gradient_descent.py`, `grid_search.py`, `helpers.py` — small helpers used by `implementations.py`
- `run.py` — tiny script to verify imports/run locally

## How to run the public tests (local path)
From the **course repo** folder `projects/project1`:
```bash
python -m pytest . -q --github_link "C:\path\to\your\repo" -k "not test_github_link_format"
