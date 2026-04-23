# Fintech Lenders in the U.S. Mortgage Market

**Authors:** Blessed C. Anaso, Patricia Nangole-Meyer
**Course:** Banks & the Fintech Revolution — University of Basel, Spring 2025
**Instructor:** Prof. Kumar Rishabh (Ph.D.)
**Date:** June 2025

An empirical study of fintech vs. traditional mortgage lenders in the United States, using Freddie Mac Single-Family Loan-Level data. We compare borrower risk profiles, interest rate structures, delinquency performance, and the predictive power of borrower metrics across the two lender types.

## Research Questions

1. How do fintech and non-fintech mortgage lenders differ in borrower selection, loan pricing, and delinquency outcomes?
2. Do fintech lenders genuinely extend credit to underserved borrowers, as the literature claims?
3. Which borrower metrics best predict delinquency, and does predictive accuracy differ between lender types?

## Data

Freddie Mac Single-Family Loan-Level Dataset — 2017 origination and servicing files, merged and cleaned down to 28,167 loans across 14 numerical variables. Sentinel values for missingness (9999 for FICO, 999 for DTI/CLTV/LTV) were filtered, skewed variables were log-transformed, and a delinquency flag was engineered from the servicing file.

The raw data is publicly available from Freddie Mac but is not committed to this repo — see `data/README.md`.

## Methods

- **Descriptive analysis** — class shares, FICO distributions by risk band (super-prime / prime / near-prime / sub-prime following Toh 2023), loan-size distributions, servicer concentration, and state-level market share.
- **Delinquency analysis** — overall and lender-stratified delinquency rates.
- **Random Forest classifiers** for delinquency prediction:
  - Model A: FICO score only — AUC 0.73
  - Model B: DTI ratio only — AUC 0.61
- **Stratified interest-rate models** trained separately on fintech and non-fintech subsamples — AUC 0.71 (fintech) vs. 0.64 (non-fintech).

## Key Findings

- **Market share.** Fintechs originated ~14.4% of loans in the sample, consistent with BIS (2020) estimates.
- **Borrower profiles.** Fintech borrowers had lower average FICO scores (733.7 vs. 748.5) and were nearly twice as likely to fall in the near-prime band (8.9% vs. 4.7%) — consistent with a financial-inclusion narrative, though the absence of income, race, and first-time-buyer flags prevents us from confirming it directly.
- **Pricing.** Fintech interest rates were marginally higher (4.20% vs. 4.14%), echoing Buchak et al. (2018).
- **Delinquency.** Fintech loans went delinquent *more* often (11.9% vs. 9.9%), contradicting Fuster et al. (2019) and Cornelli et al. (2020) who argue that fintech screening technology should lower default rates.
- **Risk-based pricing.** Interest-rate models predicted delinquency better for fintechs (AUC 0.71) than for traditional lenders (AUC 0.64), suggesting fintech rates reflect true risk more accurately.
- **Servicing.** Fintech loans are overwhelmingly serviced by Quicken Loans, pointing to a vertically integrated model; non-fintech servicing is fragmented across many third parties.

## Repository Structure

```
.
├── README.md
├── LICENSE
├── .gitignore
├── paper/
│   └── Anaso_Meyer_2025_Fintech_Lenders_Mortgage_Market.pdf
├── code/
│   └── fintech_mortgage_analysis.py   # Full Python pipeline
├── data/
│   └── README.md                      # Freddie Mac download instructions
└── figures/                           # Plots exported from the script
```

## Reproducing the Analysis

**Requirements:** Python 3.10+, with `numpy`, `pandas`, `matplotlib`, `seaborn`, `plotly`, and `scikit-learn`. A `requirements.txt` is included.

```bash
git clone https://github.com/<your-username>/fintech-mortgage-lending.git
cd fintech-mortgage-lending
pip install -r requirements.txt
```

1. Download the 2017 Freddie Mac origination and servicing files (see `data/README.md`).
2. Place them in `data/` as `origination2017.csv` and `servicing2017.csv`.
3. Run `python code/fintech_mortgage_analysis.py`.

The script was originally developed in a Jupyter notebook; cell boundaries (`# In[..]:`) are preserved in the `.py` export for readability.

## Author Contributions

Joint project between Blessed C. Anaso and Patricia Nangole-Meyer. Both authors contributed to the research design, literature review, and written analysis. Code, data cleaning, and modeling were developed collaboratively.

## License

Released under the MIT License. See `LICENSE`. The Freddie Mac data is governed by its own terms of use.

## Contact

Blessed C. Anaso — contact details on my CV.
