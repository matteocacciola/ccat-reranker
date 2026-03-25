import numpy as np


def litm(documents):
    """
    Function based on Haystack's LITM ranker:
    https://github.com/deepset-ai/haystack/blob/main/haystack/nodes/ranker/lost_in_the_middle.py
    Lost In The Middle is based on the paper https://arxiv.org/abs/2307.03172. Check it for mor details.

    Args:
        documents: List of documents (the declarative working memories)

    Returns:
        The same list but reordered
    """
    if len(documents) == 1:
        return documents
    
    document_index = list(range(len(documents)))
    lost_in_the_middle_indices = [0]

    litm_docs = []
    for doc_idx in document_index[1:]:
        insertion_index = len(lost_in_the_middle_indices) // 2 + len(lost_in_the_middle_indices) % 2
        lost_in_the_middle_indices.insert(insertion_index, doc_idx)
        litm_docs.extend([documents[idx] for idx in lost_in_the_middle_indices])
    return litm_docs


def sbert_ranker(documents, query, model):
    sentence_combinations = [[query, document[0].page_content] for document in documents]
    scores = model.predict(sentence_combinations)

    ranked_indices = np.argsort(scores)[::-1]
    out_list = [documents[idx] for idx in ranked_indices]
    # I don't change the score in the Documents using the reranker score 
    # because it could be very different from the classical bi-encoder and could create mistakes
    return out_list
