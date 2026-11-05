# Related work notes (read 2026-11-05)

Notes on the closest prior work, used to position the paper. "Assumes" lists what the method needs to work.

## Stale-Document Poisoning (Ahmed, Galke Poech, Röttger; arXiv 2609.31342, Sep 2026)
- **Task:** does outdated retrieved evidence overturn an answer the model gets right closed-book? "Selective trust: whether retrieved evidence still applies."
- **Data:** 317 knowledge reversals (medicine 87, law 100, software/API 60, platform policy 70), grounded in dated official sources; old and new documents share a format. Automated provenance and validity checks; a 50-item temporal subset.
- **Models:** 12 (GPT-4o/4.1, Claude Opus 4.5 / Sonnet 4.6, DeepSeek V3/R1, Qwen2.5 7B/72B, Llama-3.1 8B/70B, MedGemma). Dense retriever MiniLM-L6.
- **Metrics:** poisoning rate (closed-book correct → wrong with an outdated document) of 17–91% for open models; up-to-date evidence followed in 97–100% of cases.
- **Recency re-ranker:** 0.5·cos + 0.5·normalised year + 0.1·explicit supersession cue. It gains 4.6–10.0 points with correct dates. **With dates absent the gains are small and not significant; with wrong dates, selection of the current document falls to zero.**
- **Statistics:** Wilson CIs, item-clustered bootstrap (10k), exact McNemar, BH/Holm corrections. The GPT-4o judge reaches 92.5% against 200 human labels (κ 0.93 across judges).
- **Assumes:** document dates, when available; no lineage inference; no agent.
- **Relation to us:**
  - their closed-book poisoning measure → our D1;
  - their wrong-date result motivates lineage inference, so we cite it for the date-noise argument;
  - our B2 sweep extends it to lineage-based fixes.

## HoH (Ouyang, Pan, Cheng, Yan, Luo, Lin, Liu; arXiv 2503.04800, 2025)
- **Data:** Wikipedia monthly snapshot diffs → a factual-change classifier (2k human labels) → LLM-generated QA with current and outdated answers plus evidence. 96,124 QA pairs and 219,463 documents; 10k evaluated.
- **Validation:** 500 QA pairs checked by hand (98.4–100% accuracy); judge checked by hand.
- **Metrics:** a perfect/missing/harmful score; top-k hits for relevant vs. outdated passages. Point estimates only, no CIs.
- **Key retrieval finding:** "models that perform better at retrieving relevant information also show proportionally better performance at retrieving outdated information"
  - bge-base: outdated@5 0.529;
  - bge-m3: outdated@5 0.551;
  - BM25 without decay: outdated@5 0.884 vs. relevant@5 0.871.

  → **We must cite this for "better retriever ⇒ more stale"**, not claim it as new.
- **Mitigations:** time-decay ranking (Gaussian), MRAG, TempRALM, time-aware prompts; none solves it.
- **Assumes:** document modification times; outdated passages are near-copies of earlier article versions.

## MemStrata / Temporal Validity in Retrieval Memory (arXiv 2606.26511, Jun 2026, anonymised draft)
- A memory layer for agents: an extracted (subject, relation) key whose new value closes the old fact's validity interval. Deterministic on the read path.
- Benchmarks of 20–50 items. Naive RAG serves the superseded value 15–40% of the time; MemStrata about 0%.
- **Warns that benchmarks with textual markers ("deprecated") let baselines read the label; removing markers moved accuracy by up to 14 points** → supports our declared/masked/strict settings.
- **Assumes:** ingestion order stands for time; extraction succeeds (only 44% on a messier set).

## GVD (Sediqin, Dalmia, Thoppanahalli, Mukherji; Centific; arXiv 2609.17696, Sep 2026)
- Version families plus rule-level relations (duplicate, contradiction, asymmetric refinement, neutral). Uses MiniLM, DeBERTa NLI and counterfactual span probing; **no LLM**.
- 120 confidential enterprise documents (140 ingestions, 59 families). 300 + 500 expert-labelled pairs (κ 0.88 / 0.85). Family F1 0.971; rule F1 0.94.
- Baselines: MinHash, SimHash, SBERT, RETSim (0.943).
- **No downstream retrieval or RAG evaluation; data not public.**
- Closest to our lineage inference. Ours differs in being public, in directed supersession with chains, in measuring downstream RAG effect, and in having an agent stage.

## ChurnBench (Singh, Priyam, Bhowmick; arXiv 2609.11515, Sep 2026)
- Freshness errors of agents over enterprise data sources with an append-only ground-truth ledger. Refresh scheduling (TTL), not cache age, governs staleness.
- **Assumes:** timestamps and change logs; it is about the data layer, not documents.

## Modani et al., WISE 2021: "Detecting Document Versions and Their Ordering in a Collection"
- Three stages: find versions, then order them. Wikipedia-derived dataset, released (github natwar-modani/versions). Beats near-duplicate detection.
- → Our B3 baseline is in this style: cluster near-duplicates, then order by date.

## Older, already cited
- ConflictBank (2408.12076: misinformation, temporal and semantic conflicts).
- WikiContradict (2406.13805: 253 human-annotated real conflicts).
- TimeQA, RealTime QA, Dhingra et al. (time-sensitive QA).
- Longpre et al.; Xie et al. (knowledge conflicts).

## Positioning summary
- **Problem:** "RAG serves outdated documents" is established (HoH; Poisoning; MemStrata). Our RFC results confirm it in a new domain, with a real lineage graph (chains, splits) instead of edit diffs or synthetic reversals.
- **Fixes in prior work need reliable time metadata** (time decay, a recency re-ranker, ingestion order, TTL). Poisoning shows the re-ranker collapses with wrong dates.
- **Our contribution:** infer *lineage* (who replaces whom), not just recency, with leakage control, a public benchmark, and a cascade whose second stage is a tool-using agent. We measure the downstream effect on retrieval and answers.
- **Rigour bar to meet:** Poisoning's statistics (Wilson/bootstrap CIs, McNemar, multiple-comparison correction; judge checked against 200 human labels) and HoH's human checks (500 QA). We match the statistics; human validation is our gap (E1).
