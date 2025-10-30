# OpenADMET and ExpansionRx Blind Challenge

Code for entry into the [ OpenADMET + ExpansionRx Blind Challenge](https://huggingface.co/spaces/openadmet/OpenADMET-ExpansionRx-Challenge).

## Performances on limited test set

### Initial submission - Chemprop Multitask Predictions

| Endpoint     | MAE | R2 | Spearman R | Kendall's Tau |
| ---          | --- | --- | --- | --- |
| Overall      | 0.79 +/- 0.03 | 0.21 +/- 0.06 | 0.63 +/- 0.03 | 0.47 +/- 0.02 |
| LogD         | 0.49 +/- 0.01 | 0.62 +/- 0.02 | 0.81 +/- 0.01 | 0.63 +/- 0.01 |
| KSOL         | 0.43 +/- 0.01 | 0.48 +/- 0.03 | 0.64 +/- 0.02 | 0.46 +/- 0.02 |
| MLM CLint    | 0.42 +/- 0.01 | 0.17 +/- 0.05 | 0.50 +/- 0.03 | 0.35 +/- 0.02 |
| HLM CLint    | 0.38 +/- 0.01 | 0.10 +/- 0.08 | 0.49 +/- 0.04 | 0.35 +/- 0.03 |
| Caco2 Efflux | 0.47 +/- 0.01 | -0.38 +/- 0.06 | 0.54 +/- 0.03 | 0.39 +/- 0.02 |
| Caco2 A>B    | 0.43 +/- 0.01 | -0.81 +/- 0.11 | 0.46 +/- 0.03 | 0.31 +/- 0.02 |
| MPPB         | 0.21 +/- 0.01 | 0.55 +/- 0.05 | 0.74 +/- 0.03 | 0.56 +/- 0.03 |
| MBPB         | 0.21 +/- 0.01 | 0.57 +/- 0.04 | 0.77 +/- 0.03 | 0.57 +/- 0.03 |
| MGMB         | 0.18 +/- 0.01 | 0.60 +/- 0.07 | 0.76 +/- 0.04 | 0.60 +/- 0.04 |

## Local development

### Prerequisites

- [Conda](https://docs.conda.io/projects/conda/en/latest/user-guide/install/download.html)

### Create Environment

The following commands will setup an environment where you can run and test the application locally:

```bash
git clone git@github.com:jonswain/OpenADMET_ExpansionRx_Blind_Challenge
cd OpenADMET_ExpansionRx_Blind_Challenge
conda env create -f environment.yml
conda activate OpenADMET_challenge
code .
```
