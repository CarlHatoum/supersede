# Labelling guide: validating the answer judge

> **Status (2026-11-03).** Claude Opus 5.5 labelled all 100 rows blind, as a second *model* rater: `results/rfc_answer_staleness_hybrid_labels_claude-opus-5-5.csv`, with notes on borderline rows. Agreement with the Haiku judge is **κ = 0.846 (91/100)**, or κ = 0.746 (94/100) on the reported outcome buckets. All 3 clear `stale` cases agree.
>
> This is model–model agreement, not human validation. The human step is now a **29-row spot check**, `results/rfc_answer_staleness_hybrid_human_check.csv`: the 9 judge/Claude disagreements plus 20 random rows, shuffled, with both model labels hidden. It takes about 15 minutes. Run `--kappa` on that file to get human-vs-judge agreement.

The file is `results/rfc_answer_staleness_hybrid_to_label.csv`, with 100 answers in a seeded random order. Open it in Numbers, Excel or Google Sheets. For each row, read `question`, the two reference answers and `answer`, then write **one** label in `human_label`:

| label | when |
|---|---|
| `current` | Gives the current reference answer and doesn't present the superseded one as valid. |
| `stale` | Gives the superseded reference answer as if it were still in force. |
| `mixed_flagged` | Gives the current answer *and* mentions the superseded one, clearly marked as previous or replaced. |
| `mixed_unflagged` | Gives both without making clear which one is in force. |
| `abstain` | Says the documents don't contain the information. |
| `other` | Matches neither reference (wrong, or off-topic). |

Judge by meaning, not wording. Partly correct answers go to the closest label, with a note in a spare column if useful. The judge's labels are deliberately left out of the file. Don't look them up before you finish.

Save as CSV (UTF-8) under the same name, then run:

```bash
.venv/bin/python -m supersede.experiments.answer_staleness --kappa results/rfc_answer_staleness_hybrid_to_label.csv
```

This reports Cohen's κ and raw agreement between you and the judge. The bar is **κ ≥ 0.6** before the judge's numbers are published. If it falls short, read the disagreements and revise the judge prompt (j-v2). This takes about 45–60 minutes. A second rater on the same file gives inter-human κ, which strengthens the paper.
