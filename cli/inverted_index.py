from __future__ import annotations

import os
import pickle
import re
from collections import Counter
from pathlib import Path
from typing import Dict, Mapping, Set
import math 

BM25_K1 = 1.5
BM25_B = 0.75
class InvertedIndex:
    def __init__(
        self,
        index: Dict[str, Set[str]] | None = None,
        term_frequencies: Counter[tuple[str, str]] | None = None,
        docmap: Dict[str, str] | None = None,
        doc_lengths: Dict[str, int] | None = None,
    ) -> None:
        self.index = index if index is not None else {}
        self.term_frequencies = term_frequencies if term_frequencies is not None else Counter()
        self.docmap = docmap if docmap is not None else {}
        self.doc_lengths = doc_lengths if doc_lengths is not None else {}
        

    def tokenize(self, text: str) -> list[str]:
        return re.findall(r"\w+", text.lower())

    def _add_document(self, doc_id: str, text: str) -> None:
        tokens = self.tokenize(text)
        for token in tokens:
            self.index.setdefault(token, set()).add(doc_id)


        # Store per-document term counts without wiping previous docs.
        for token, count in Counter(tokens).items():
            self.term_frequencies[(doc_id, token)] = count
        self.doc_lengths[doc_id] = len(tokens)

    def get_documents(self, term: str) -> list[str]:
        tokens = self.tokenize(term)
        if not tokens:
            return []
        return sorted(self.index.get(tokens[0], set()))

    def build(self, documents: Mapping[str, str]) -> None:
        for raw_doc_id, text in documents.items():
            doc_id = str(raw_doc_id)
            self.docmap[doc_id] = text
            self._add_document(doc_id, text)

    def save(self, index_path: str, docmap_path: str, term_frequencies_path: str, doc_lengths_path: str) -> None:
        for path in (index_path, docmap_path, term_frequencies_path , doc_lengths_path):
            parent = Path(path).parent
            if parent and not parent.exists():
                os.makedirs(parent, exist_ok=True)

        with open(index_path, "wb") as f:
            pickle.dump(self.index, f)
        with open(docmap_path, "wb") as f:
            pickle.dump(self.docmap, f)
        with open(term_frequencies_path, "wb") as f:
            pickle.dump(self.term_frequencies, f)
        with open(doc_lengths_path, "wb") as f:
            pickle.dump(self.doc_lengths, f)

    def load(self, index_path: str, docmap_path: str, term_frequencies_path: str, doc_lengths_path: str) -> None:
        with open(index_path, "rb") as f:
            self.index = pickle.load(f)
        with open(docmap_path, "rb") as f:
            self.docmap = pickle.load(f)
        with open(term_frequencies_path, "rb") as f:
            self.term_frequencies = pickle.load(f)
        with open(doc_lengths_path, "rb") as f:
            self.doc_lengths = pickle.load(f)

    def get_tf(self, doc_id: str, term: str) -> int:
        tokens = self.tokenize(term)
        if len(tokens) != 1:
            raise ValueError("Term must be a single token")
        return int(self.term_frequencies.get((str(doc_id), tokens[0]), 0))
    def get_idf(self , term:str) -> float : 
        tokens = self.tokenize(term)
        if len(tokens) !=1 : 
            raise ValueError("term must be a single token") 
        term_match_doc_count = len(self.get_documents(tokens[0]))
        return math.log((len(self.docmap) + 1) / (term_match_doc_count + 1))
    def get_bm25_idf(self, term: str) -> float : 
        tokens = self.tokenize(term)
        if len(tokens) != 1:
            raise ValueError("Term must be a single token")
        term_match_doc_count = len(self.get_documents(tokens[0]))
        return math.log((len(self.docmap) - term_match_doc_count + 0.5) / (term_match_doc_count + 0.5) + 1)
    def get_bm25_tf(self, doc_id, term, k1=BM25_K1 , b=BM25_B) -> float :
        average_doc_length = self.__get_avg_doc_length()
        doc_length = self.doc_lengths.get(str(doc_id), 0)
        length_norm = (1 - b) + b * (doc_length / average_doc_length) if average_doc_length > 0 else 1
        tokens = self.tokenize(term)
        if len(tokens) != 1:
            raise ValueError("Term must be a single token")
        tf = self.get_tf(doc_id, term)
        tf = tf * (k1 + 1) / (tf + k1 * length_norm) if tf > 0 else 0.0
        return tf
    def __get_avg_doc_length(self) -> float : 
        total_length = sum(self.doc_lengths.values())
        return total_length / len(self.doc_lengths) if self.doc_lengths else 0.0
    def bm25(self, doc_id, term) -> float : 
        idf = self.get_bm25_idf(term)
        tf = self.get_bm25_tf(doc_id, term)
        return idf * tf
    def bm25_search(self, query, limit) : 
        tokens = self.tokenize(query)
        scores = Counter()
        for token in tokens:
            for doc_id in self.get_documents(token):
                scores[doc_id] += self.bm25(doc_id, token)
        ranked = sorted(scores.items(), key=lambda item: (-item[1], item[0]))
        return ranked[:limit]
    def get_document_title(self, doc_id: str) -> str:
        text = self.get_document_text(doc_id)
        lines = text.split('\n')
        if lines:
            return lines[0]
        return ""
    def get_document_text(self, doc_id: str) -> str:
        return self.docmap.get(str(doc_id), "")
    

