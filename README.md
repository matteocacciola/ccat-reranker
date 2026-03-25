# Grinning Cat ReRanker

[![awesome plugin](https://custom-icon-badges.demolab.com/static/v1?label=&message=awesome+plugin&color=F4F4F5&style=for-the-badge&logo=cheshire_cat_black)](https://)

**Grinning Cat ReRanker** applies post-recall reranking to the **declarative memory** using two complementary strategies that can be enabled independently or combined:

- **SBERT Cross-Encoder** — scores each retrieved document against the user query using a `sentence-transformers`
  [CrossEncoder](https://www.sbert.net/docs/cross_encoder/usage/usage.html) model, then reorders documents by relevance.
  The default model is `cross-encoder/ms-marco-MiniLM-L-6-v2`, but any compatible HuggingFace cross-encoder can be used.
- **Lost In The Middle (LITM)** — reorders documents following the method described in the
  [Lost in the Middle paper](https://arxiv.org/abs/2307.03172), placing the most relevant documents at the beginning and
  end of the context window to mitigate the LLM tendency to ignore content in the middle. When SBERT is also enabled,
  LITM is applied on top of the cross-encoder output.

> **Note:** Only the **declarative memory** is affected. Episodic and procedural memories are not reranked.

## Settings

| Parameter  | Type   | Default                                | Description                            |
|------------|--------|----------------------------------------|----------------------------------------|
| `litm`     | `bool` | `False`                                | Enable Lost In The Middle reranking    |
| `sbert`    | `bool` | `False`                                | Enable SBERT cross-encoder reranking   |
| `ranker`   | `str`  | `cross-encoder/ms-marco-MiniLM-L-6-v2` | HuggingFace cross-encoder model to use |

**Possible combinations:**

| `sbert`  | `litm`  | Behaviour                             |
|----------|---------|---------------------------------------|
| `False`  | `False` | No reranking applied                  |
| `False`  | `True`  | LITM reranking only                   |
| `True`   | `False` | SBERT cross-encoder reranking only    |
| `True`   | `True`  | SBERT cross-encoder, then LITM on top |

> **Tip:** Rerankers, especially LITM, are most effective when the memory retrieval returns at least 10 documents.
> Consider increasing the `k` parameter of the declarative memory retrieval in the Grinning Cat settings.

## Requirements

- `sentence-transformers >= 4.1.0`
- `unstructured >= 0.18.1`

## Installation

1. Navigate to the **Plugins** page in the Grinning Cat admin panel.
2. Find the **Grinning Cat ReRanker** plugin.
3. Click **Install**.

## Configuration

1. Navigate to the **Plugins** page.
2. Find the **Grinning Cat ReRanker** plugin.
3. Click the **Settings** icon (bottom-right of the plugin card).
4. Enable/disable `litm` and/or `sbert`, and optionally change the `ranker` model.
5. Save your settings.

## Author

[nickprock](https://github.com/nickprock) — plugin repository: [matteocacciola/ccat-reranker](https://github.com/matteocacciola/ccat-reranker)
