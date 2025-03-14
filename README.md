# Federated Learning for Large Language Models

Currently we have implemented some baseline approaches for domain-specific fine-tuning of language models using LoRA (Low-Rank Adaptation) in a federated manner. The project enables training specialized models for different domains while evaluating cross-domain performance.

## Overview

This mid-way project consists of three main components:

1. **Data Allocation**: Partitioning datasets by category into training and test sets.
2. **Baseline: Domain-Specific Fine-Tuning**: Training LoRA adapters for each domain.
3. **Model Evaluation**: Comparing fine-tuned models against the base model using GPT-4 as an evaluator.


## Usage

### 0. Install Requirements

Install the required packages:

```bash
pip install -r requirements.txt
```

### 1. Data Allocation

Partition a dataset into domain-specific training and test sets:

```bash
python client_data_allocation.py
```

### 2. Baseline: Domain-Specific Fine-Tuning

Train LoRA adapters for specific domains:

```bash
python baseline_training.py --domain brainstorming classification summarization
```

### 3. Model Evaluation

Evaluate fine-tuned models against the original model using GPT-4:

```bash
python evaluation.py --domain brainstorming classification summarization

```

## Notes

- The evaluation uses GPT-4 as an unbiased judge to score model outputs on a scale of 1-10.
- Fine-tuning is performed using LoRA to efficiently adapt models for specific domains.
- Each domain's model is evaluated against all other domains to measure generalization.

## Future Work

This repository currently implements a baseline approach. Future work will include:
- Proposed federated learning approach for cross-domain knowledge sharing
- Model aggregation strategies
- Privacy-preserving techniques
- Advanced evaluation metrics