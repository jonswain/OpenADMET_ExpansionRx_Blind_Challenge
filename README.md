# OpenADMET and ExpansionRx Blind Challenge

Code for entry into the [ OpenADMET + ExpansionRx Blind Challenge](https://huggingface.co/spaces/openadmet/OpenADMET-ExpansionRx-Challenge).

## Performances on limited test set

### Initial submission - with error in code causing significant outliers (inf values)

Impressively bad

| Endpoint | MAE | R2 | Spearman R | Kendall's Tau |
| --- | --- | --- | --- | --- |
| Overall | 2.59 +/- 0.10 | -19.87 +/- 1.05 | 0.34 +/- 0.04 | 0.26 +/- 0.03 |
| LogD | 0.95 +/- 0.03 | -0.48 +/- 0.09 | 0.30 +/- 0.03 | 0.23 +/- 0.02 |
| KSOL | 2.11 +/- 0.07 | -14.99 +/- 0.75 | 0.28 +/- 0.03 | 0.20 +/- 0.02 |
| MLM CLint | 0.95 +/- 0.02 | -3.04 +/- 0.30 | 0.11 +/- 0.04 | 0.08 +/- 0.02 |
| HLM CLint | 0.60 +/- 0.02 | -1.17 +/- 0.17 | -0.07 +/- 0.05 | -0.04 +/- 0.03 |
| Caco2 Efflux | 0.47 +/- 0.01 | -0.29 +/- 0.05 | 0.32 +/- 0.03 | 0.22 +/- 0.02 |
| Caco2 A>B | 4.02 +/- 0.10 | -159.81 +/- 7.83 | 0.07 +/- 0.04 | 0.05 +/- 0.03 |
| MPPB | 0.39 +/- 0.01 | -0.33 +/- 0.12 | 0.56 +/- 0.04 | 0.39 +/- 0.03 |
| MBPB | 0.19 +/- 0.01 | 0.64 +/- 0.03 | 0.76 +/- 0.03 | 0.57 +/- 0.03 |
| MGMB | 0.18 +/- 0.01 | 0.61 +/- 0.07 | 0.77 +/- 0.04 | 0.60 +/- 0.04 |



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
