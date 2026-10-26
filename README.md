# Retrieval-Augmented Generation for Recipes and Scientific Literature

**H02C8b — Information Retrieval and Search Engines**

> **Project report:** [H02C8b Information Retrieval and Search Engines: Project Handout (PDF)](IRSE-project.pdf)

## Abstract

This repository investigates how document representation, retrieval strategy, and prompt design affect retrieval-augmented generation (RAG). Part 1 implements a sparse TF–IDF retriever over a recipe collection and uses retrieved recipes to condition an instruction-tuned language model. Part 2 studies retrieval over ACL Anthology papers, comparing lexical and dense representations, long-document chunking, query rewriting, and Hypothetical Document Embeddings (HyDE). Separate prompt-engineering experiments examine citation compliance, abstention, and resistance to instructions embedded in retrieved material.

The accompanying report finds that dense retrieval improves over the lexical baseline on scientific papers, structure-aware chunking improves document-level ranking, and HyDE outperforms direct query embedding. Query rewriting, however, reduces retrieval effectiveness in this setting. These are findings from the reported experiments, not claims of general superiority across datasets.

## Research questions

1. How do field selection, vocabulary construction, and retrieval depth affect a lexical RAG baseline?
2. Does dense retrieval improve scientific-paper retrieval relative to TF–IDF?
3. Can passage-level representations mitigate the limitations of embedding long documents?
4. Do language-model-generated rewrites or hypothetical abstracts improve retrieval?
5. To what extent do explicit prompting constraints support grounded answers, valid citations, and abstention?

## Methodology

### Part 1 — Recipe retrieval and generation

The lexical pipeline is implemented in [`part_1/tf_idf.py`](part_1/tf_idf.py), with generation experiments in [`main.py`](main.py).

- **Indexed fields:** recipe name, tags, description, and ingredients. Cooking steps are excluded from retrieval after a field-selection experiment, but included in the generation context to provide procedural evidence.
- **Preprocessing:** lowercase alphabetic tokenization, English stopword removal, WordNet noun lemmatization, and corpus-trained bigram detection with Gensim (`min_count=5`, `threshold=3.0`). Numeric information is discarded by tokenization.
- **Vocabulary:** document-frequency filtering with `min_df=5`, `max_df_ratio=0.5`, and a maximum of 50,000 terms in the main pipeline. The field-selection script uses the helper's default `min_df=6` and trains its bigram model on all fields, so its scores are not directly interchangeable with the final evaluation.
- **Weighting:** logarithmic term frequency, smoothed inverse document frequency, and L2 normalization:

  ```text
  tf(t, d)  = 1 + ln(count(t, d))
  idf(t)    = ln((N + 1) / (df(t) + 1)) + 1
  w(t, d)   = tf(t, d) × idf(t)
  score(q,d) = cosine(q, d)
  ```

- **Retrieval:** exhaustive sparse-vector scoring; the default evaluation depth is 10 documents, while generation uses 5 recipes. Queries with no in-vocabulary terms return no results.
- **Generator:** [`mistralai/Mistral-7B-Instruct-v0.2`](https://huggingface.co/mistralai/Mistral-7B-Instruct-v0.2), loaded with 4-bit NF4 quantization and the Mistral `[INST] ... [/INST]` prompt format.

Generation experiments compare minimal instructions, structured grounding instructions, and structured instructions augmented with retrieval scores. Additional tests examine irrelevant context, reasoning across recipes, and responses with versus without retrieved evidence.

### Part 2 — Scientific-paper retrieval

The scientific-paper experiments are coordinated by [`part_2/part2.py`](part_2/part2.py).

| Component | Experimental design |
| --- | --- |
| Lexical baseline | TF–IDF over paper titles and abstracts |
| Dense baseline | Normalized embeddings from [`sentence-transformers/all-mpnet-base-v2`](https://huggingface.co/sentence-transformers/all-mpnet-base-v2), ranked by cosine similarity |
| Field selection | Title only; title + abstract; title + abstract + truncated full text |
| Fixed-window retrieval | 220-word windows with 40-word overlap; maximum chunk score per paper |
| Structure-aware retrieval | Section/paragraph boundaries with sentence-group fallback; weighted aggregation of the top 3 chunk scores |
| Query rewriting | One rewrite, or the original query and up to 3 rewrites combined using reciprocal rank fusion (`k_rrf=60`) |
| HyDE | Generate a hypothetical abstract, concatenate it with the original query, and embed the combined text for retrieval |
| Prompt controls | Allowed ACL-ID citations, explicit abstention, and instructions to treat retrieved text as evidence rather than executable instructions |

The chunking experiments are evaluated separately from rewriting and HyDE: the latter use the title-and-abstract dense baseline, not a combined chunking-plus-HyDE system. The hypothetical abstract is an intermediate retrieval representation, not a factual answer.

## Evaluation and reported findings

**Provenance:** The following values are transcribed from the [project report](IRSE-project.pdf), particularly Tables 4 and 8–12. They have not been independently reproduced as part of this documentation update.

### Evaluation protocol

- **Recipes:** 47 supplied queries, of which 41 have relevant documents and are included in retrieval evaluation.
- **ACL Anthology:** 98 supplied queries, all with relevance judgments.
- **Metrics:** macro- and micro-averaged precision, recall, and F1, plus mean average precision over the returned ranking. The default ranking depth is 10.
- **Relevance:** binary membership in the supplied relevant-document sets; recipe relevance annotations are reduced to document IDs.
- **Average precision:** the sum of precision at relevant ranks is divided by the total number of known relevant documents, including those not retrieved. Part 2 labels the resulting mean over top-10 rankings `MAP@10`.

### Recipe baseline

| Metric | Macro | Micro |
| --- | ---: | ---: |
| Precision | 0.0805 | 0.0805 |
| Recall | 0.1148 | 0.0570 |
| F1 | 0.0798 | 0.0667 |

**MAP over the top-10 ranking: 0.0618.** Retrieval-depth experiments show that `k=3` maximizes macro-F1 among the tested values, while MAP increases with larger rankings. The implementation uses `k=10` for evaluation and `k=5` for generation as separate operating points.

### ACL Anthology retrieval

| System | Macro F1 | Micro F1 | MAP@10 |
| --- | ---: | ---: | ---: |
| TF–IDF, title + abstract | 0.1508 | 0.1572 | 0.3727 |
| Dense baseline, title + abstract | 0.1709 | 0.1736 | 0.4491 |
| Fixed windows + maximum score | 0.1815 | 0.1858 | 0.4649 |
| Structure-aware chunks + weighted top-3 | 0.1950 | 0.1997 | 0.4886 |
| Dense baseline + single query rewrite | 0.1491 | 0.1510 | 0.3933 |
| Dense baseline + RRF-3 rewriting | 0.1655 | 0.1684 | 0.4204 |
| Dense baseline + HyDE | **0.2059** | **0.2118** | **0.5477** |

HyDE yields the highest reported MAP@10 among these configurations, approximately 22% above the dense baseline. Structure-aware chunking improves MAP@10 by approximately 8.8%. Query rewriting underperforms the original query; the report identifies semantic drift as a possible explanation, rather than an experimentally isolated cause.

### Prompt behavior

The report records successful outcomes for **2/2 inline-citation tests**, **3/3 abstention tests**, and **2/2 injection-hardening tests**. These experiments use fixed, manually selected papers to isolate prompt behavior from retrieval errors. They are small compliance checks, not end-to-end answer-quality benchmarks or a general security guarantee. Citation-format and allowed-ID checks do not establish that every claim is supported by its citation.

## Repository structure

```text
.
├── IRSE-project.pdf                 # Academic report, analyses, and bibliography
├── README.md
├── main.py                         # Model loading and both experiment suites
├── requirements.txt
├── part_1/
│   ├── tf_idf.py                   # Preprocessing, vocabulary, sparse vectors
│   ├── metrics.py                  # Retrieval and evaluation helpers
│   ├── evaluate.py                 # Baseline, threshold, and k sensitivity
│   ├── field_experiment.py         # Recipe field-selection experiment
│   └── irse_queries_2026_recipes.json
└── part_2/
    ├── part2.py                    # Scientific-paper experiment orchestration
    ├── part2_data.py               # Sample construction and data loading
    ├── part2_metrics.py            # ACL-ID-based retrieval evaluation
    ├── neural_retrieval.py         # Dense retrieval index
    ├── long_docs.py                # Chunking and document-score aggregation
    ├── query_rewriting.py          # LM rewrites and reciprocal rank fusion
    ├── hyde.py                     # Hypothetical-abstract retrieval
    ├── prompt_engineering.py       # Citation, abstention, and hardened prompts
    └── acl_anthology_queries.json
```

## Reproduction

### Environment

Use **Python 3.10 or later**; the source uses modern union type annotations. From the repository root:

```sh
python3 -m venv venv
. venv/bin/activate
python -m pip install -r requirements.txt
```

The full generation pipeline requires a compatible **NVIDIA CUDA environment**: Part 1 explicitly moves inputs to `cuda`, and model loading uses BitsAndBytes 4-bit quantization. The full suite is not configured for CPU-only or macOS Metal execution. Retrieval-only experiments do not require loading Mistral, although dense encoding and full-text chunking benefit from acceleration.

First use may download NLTK stopwords and WordNet resources and model weights from Hugging Face. Network access, sufficient model-cache storage, and any authentication required by the model host must be available. Generation runs locally after model loading; no hosted inference API is configured.

### Data preparation

Parquet files are excluded by [`.gitignore`](.gitignore) and are **not included in this checkout**. Obtain the corresponding course datasets separately and place them as follows:

| File | Purpose |
| --- | --- |
| `part_1/irse_documents_2026_recipes.parquet` | Required recipe collection |
| `part_2/acl_anthology_full.parquet` | Full scientific-paper collection used to construct the sample |
| `part_2/acl_anthology_queries.parquet` | Query-associated papers to include in the sample |
| `part_2/anthology_sample.parquet` | Generated sample cache; an existing matching cache can be reused |

The JSON query files are already included. Recipe judgments refer to dataset row indices, so preserve the original recipe ordering. ACL judgments refer to `acl_id` values.

[`part_2/part2_data.py`](part_2/part2_data.py) constructs the sample by concatenating the query-associated papers with 2,000 papers sampled from the full collection, then shuffling with seed `1072381`. If the query-associated Parquet file is absent, the code falls back to sampling only from the full collection. **That fallback may omit relevant papers and should not be treated as equivalent to the reported evaluation.** An existing sample cache is reused without rebuilding; ensure it matches the intended dataset configuration.

### Run experiments

The commands below assume the virtual environment is active and the shell starts in the repository root. Explicit `PYTHONPATH` settings accommodate the current mix of package-relative and sibling-module imports without modifying the source.

**Recipe retrieval evaluation and field selection — no language model required:**

```sh
(cd part_1 && PYTHONPATH=.. python -m part_1.evaluate)
(cd part_1 && PYTHONPATH=.. python -m part_1.field_experiment)
```

These scripts resolve their input files relative to the working directory, hence execution from `part_1/`.

**Scientific-paper data checks and retrieval-only experiments:**

```sh
python part_2/part2.py --step data
python part_2/part2.py --step neural
python part_2/part2.py --step long
```

**Full recipe and scientific-paper experiment suites, including generation:**

```sh
PYTHONPATH=part_2 python main.py
```

Use this full-suite entry point for rewriting, HyDE, and prompt-engineering experiments: it loads and passes the model and tokenizer. The standalone Part 2 `--step all` path does not load them and is not a complete reproduction command. Experiment outputs are printed to standard output.

## Discussion

The recipe baseline illustrates the limits of lexical matching: spelling errors, vocabulary mismatch, numeric constraints, and reasoning-oriented questions are not reliably handled. Dense embeddings improve the scientific-paper baseline but remain sensitive to representation choice and encoder input limits. Chunking helps expose localized evidence, while query-side generation can either improve matching, as with HyDE here, or introduce drift.

Retrieval effectiveness and answer faithfulness are distinct evaluation targets. Comparing responses with and without context can suggest use of retrieved evidence, but does not prove that an answer relies exclusively on it. Likewise, excluding queries without relevant documents from retrieval metrics leaves unanswerable-query behavior to separate analysis.

The current implementation has no dedicated index for authors, publication dates, or venues, and no coauthorship graph. Metadata filtering and author-relationship reasoning are discussed as extensions in the report, not implemented capabilities.
