# Enhanced Navigation Environment: Course Project for Intelligent Agent Navigation

## Abstract
This repository contains a course project developed to study and demonstrate intelligent agent navigation in a forest-like environment. The project combines reinforcement learning (with an emphasis on Deep Q-Networks, DQN) with Large Language Model (LLM) components as a required part of the system design. The codebase includes supporting utilities for qualitative visualization and quantitative assessment.

## Key Features
- Modular navigation environments suitable for course-level experimentation
- Scripts for training and evaluating DQN-based agents
- Graphical User Interface (GUI) for interactive policy inspection
- LLM integration as an integral component of the project pipeline
- TensorBoard logs for tracking training dynamics

## Repository Structure
- `forest_navigation_env.py`: Environment simulating forest navigation tasks (PyBullet)
- `train_dqn_random_nav.py`: Script for training DQN agents with randomized navigation objectives
- `evaluate_trained_policy_gui.py`: GUI application for qualitative and quantitative policy evaluation
- `LLM/`: Modules and scripts for LLM integration and experimentation
- `training_sessions/`: Outputs from training runs (metrics, logs, and saved artifacts)
- `env.yml`: Conda environment specification for reproducibility
- `requirements.txt`: Minimal pip dependencies for running the project

## Installation Instructions
You may set up the environment using either Conda (recommended for reproducibility) or pip.

### Option A: Conda (recommended)
```bash
conda env create -f env.yml
conda activate your_env_name  # e.g. `conda activate env-aia` (choose a name without angle brackets)
```

### Option B: pip
```bash
python -m pip install -r requirements.txt
```

## LLM Configuration (Required)
The LLM component is part of the project pipeline and requires an API key via environment variables.

1. Create a `.env` file in the project root (or set system environment variables).
2. Define at least:
   - `OPENROUTER_API_KEY`: API key used by the OpenRouter backend

Optional (recommended for robustness and reproducibility):
   - `OPENROUTER_MODEL`: model identifier (default is defined in `LLM/llm.py`)
   - `OPENROUTER_MODEL_FALLBACKS`: comma-separated model IDs to try if the primary model fails
   - `OPENROUTER_SITE_URL` and `OPENROUTER_APP_TITLE`: metadata fields for OpenRouter

> Compatibility note: the LLM module also checks `GOOGLE_API_KEY` as a fallback for older setups.

## Usage Guidelines
- To initiate DQN agent training, execute:
  ```
  python train_dqn_random_nav.py
  ```
- To perform policy evaluation via the graphical interface, run:
  ```
  python evaluate_trained_policy_gui.py
  ```
- The evaluation script integrates the LLM navigator (`LLM/llm.py`) as part of the workflow.

## Dependencies (Summary)
Core runtime dependencies are listed in `requirements.txt` and include:
- `gymnasium`, `stable-baselines3`, `pybullet`, `numpy`
- `python-dotenv`, `requests` (for the LLM/OpenRouter client)
- `opencv-python` (camera visualization) and `Pillow` (image utilities)

## Demonstration Video
[![Demo Video](https://img.youtube.com/vi/NllJUIectJA/0.jpg)](https://www.youtube.com/watch?v=NllJUIectJA)

Watch the full demo on YouTube: https://youtu.be/NllJUIectJA

## License
Please specify the license governing the use and distribution of this project.

## Acknowledgments
This project was completed as part of the "Fundamentals of AI Agents" course under the supervision of Prof. Ahmed Esmaeili. The work was conducted in collaboration with my teammates Mohammed Farhan and Mohammed Rizwan Raisulhaq. The project also acknowledges the contributions of open-source libraries and the broader academic community. 
