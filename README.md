# Graph Neural Networks for Drug–Target Affinity Prediction

[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/christybiji/gnn-dti-repurposing/blob/main/notebooks/GNN_DTI_Repurposing_v2.ipynb)

A two-branch graph neural network (molecular graph + ESM-2 protein embeddings) that predicts drug–target binding affinity from BindingDB Kd data, benchmarked against simpler baselines.

> **Result: negative.** The GNN does not beat a fingerprint + gradient-boosting baseline on unseen drugs or unseen targets. The screening output is a pipeline demonstration, not a set of findings.

## Summary

| | |
|---|---|
| **Question** | Which approved drugs might bind proteins they have not been tested against? |
| **Data** | 46,115 drug–target pairs (10,659 drugs, 1,413 targets) from BindingDB Kd via Therapeutics Data Commons; Broad Repurposing Hub as screening library |
| **Label** | pKd = 9 − log10(Kd in nM). About 50% of labels are censored ("weaker than 10 µM") and are kept |
| **Model** | GINEConv drug encoder + ESM-2 (35M) protein embedding → MLP head; Tobit censored-normal loss with learned sigma |
| **Evaluation** | Four splits (random, cold drug, scaffold, cold target); four baselines; paired bootstrap; experimental noise floor of 0.829 log units |

## Results

Single model, test sets. RMSE is observed-only; CI is the concordance index; EF is the enrichment factor at the top 1%.

| Split | GNN RMSE | Baseline RMSE | GNN CI | Baseline CI | GNN EF | Baseline EF |
|---|---|---|---|---|---|---|
| Random | 1.767 | 1.194 | 0.564 | 0.635 | 3.99 | 7.36 |
| Cold drug | 1.384 | 1.197 | 0.545 | 0.629 | 2.90 | 6.13 |
| Scaffold | 1.436 | 1.204 | 0.586 | 0.621 | 4.02 | 6.20 |
| Cold target | 1.568 | 1.319 | 0.681 | 0.776 | 2.17 | 4.63 |

Baseline = ECFP4 + ESM-2 + XGBoost. Paired bootstrap, baseline minus GNN RMSE: cold drug −0.187 [−0.203, −0.170]; cold target −0.249 [−0.273, −0.225]. Both intervals exclude zero, in the baseline's favour. A 5-seed ensemble (cold-drug RMSE 1.439) did not improve on the single model.

## What went wrong

- The first Tobit loss clamped the survival probability, which gave **zero gradient to confident false positives** on censored rows. Fixed with `torch.special.log_ndtr` and a regression test. Retraining narrowed the gap (cold-target RMSE 2.125 → 1.568) but did not close it.
- The cause of the remaining gap is **unknown**. Candidates (mean-pooled protein embeddings, training setup) are untested.

## Limitations

- The screen (9.3M untested pairs scored) is dominated by a few compounds and one target (PRSS1), with a molecular-weight correlation of +0.567 and a high PAINS rate. It should not be read as candidate drugs.
- Ensemble spread tracks error only weakly.
- Predicted binding is not efficacy. "Novel" means unmeasured in the BindingDB Kd set used here.

## Repository

```
notebooks/   full pipeline with outputs from the final run (Google Colab, GPU)
src/         Tobit loss, metrics and tests (run: pip install -r requirements.txt && pytest src/tests)
docs/        summary slides
```

## AI assistance

This project was developed with AI assistance (Anthropic's Claude), used for drafting and debugging code, explaining methods, and preparing documentation and slides. I ran the experiments and am responsible for the results and their interpretation.

## Next steps

Per-residue ESM-2 with drug–protein cross-attention, training tuning, a per-target cap on screening hits, and cluster-based bootstrapping.
