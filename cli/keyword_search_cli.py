#!/usr/bin/env python3

import argparse
import json
import string
from collections import Counter
from pathlib import Path

from inverted_index import InvertedIndex

translator = str.maketrans("", "", string.punctuation)
PROJECT_ROOT = Path(__file__).resolve().parents[1]
MOVIES_PATH = PROJECT_ROOT / "data" / "movies.json"
STOPWORDS_PATH = PROJECT_ROOT / "data" / "stopwords.txt"
INDEX_PATH = PROJECT_ROOT / "cache" / "index.pkl"
DOCMAP_PATH = PROJECT_ROOT / "cache" / "docmap.pkl"
TF_PATH = PROJECT_ROOT / "cache" / "term_frequencies.pkl"
IDF_PATH = PROJECT_ROOT / "cache" / "invert_document_frequency"
DOC_LENGTHS_PATH = PROJECT_ROOT / "cache" / "doc_lengths.pkl"

def normalize(text: str) -> str:
    return text.translate(translator).lower()


def load_documents() -> dict[str, str]:
    with open(MOVIES_PATH, "r", encoding="utf-8") as f:
        payload = json.load(f)

    documents: dict[str, str] = {}
    for movie in payload.get("movies", []):
        doc_id = str(movie.get("id"))
        title = movie.get("title", "")
        description = movie.get("description", "")
        documents[doc_id] = f"{title} {description}".strip()
    return documents

def load_movie_titles() -> dict[str, str]:
    with open(MOVIES_PATH, "r", encoding="utf-8") as f:
        payload = json.load(f)

    titles: dict[str, str] = {}
    for movie in payload.get("movies", []):
        doc_id = str(movie.get("id"))
        titles[doc_id] = movie.get("title", "")
    return titles


def load_stopwords() -> set[str]:
    with open(STOPWORDS_PATH, "r", encoding="utf-8") as f:
        return {line.strip() for line in f if line.strip()}


def main() -> None:
    parser = argparse.ArgumentParser(description="Keyword Search CLI")
    subparsers = parser.add_subparsers(dest="command", required=True, help="Available commands")

    search_parser = subparsers.add_parser("search", help="Search movies using keyword overlap")
    search_parser.add_argument("query", type=str, help="Search query")
    subparsers.add_parser("build", help="Build and cache inverted index")


    tf_parser = subparsers.add_parser("tf", help="Get term frequency for a term in a document")
    tf_parser.add_argument("doc_id", type=str, help="Document ID")
    tf_parser.add_argument("term", type=str, help="Single-token term")
    idf_parser = subparsers.add_parser("idf", help="Get inverse document frequency for a term")
    idf_parser.add_argument("term", type=str, help="Single-token term")
    #adding tfidf parser
    tfidf_parser = subparsers.add_parser("tfidf", help="Get TF-IDF score for a term in a document")
    tfidf_parser.add_argument("doc_id", type=str, help="Document ID")
    tfidf_parser.add_argument("term", type=str, help="Single-token term")
    
    bm25_idf_parser = subparsers.add_parser("bm25idf", help="Get BM25 IDF score for a term")
    bm25_idf_parser.add_argument("term", type=str, help="Single-token term")

    #adding a  bm25tf command
    #bm25_tf_parser.add_argument("b", type=float, nargs='?', default=BM25_B, help="Tunable BM25 b parameter")
    bmf25_tf_parser = subparsers.add_parser("bm25tf", help="Get BM25 TF score for a term in a document")
    bmf25_tf_parser.add_argument("doc_id", type=str, help="Document ID")
    bmf25_tf_parser.add_argument("term", type=str, help="Single-token term")
    bmf25_tf_parser.add_argument("b", type=float, nargs='?', default=0.75, help="Tunable BM25 b parameter")
    bm25search_parser = subparsers.add_parser("bm25search", help="Search movies using full BM25 scoring")
    bm25search_parser.add_argument("query", type=str, help="Search query")
    #adding limit default 5 
    bm25search_parser.add_argument("--limit", type=int, default=5, help="Limit number of results")
    


    args = parser.parse_args()
    index = InvertedIndex()

    match args.command:
        case "search":
            stop_words = load_stopwords()
            try:
                index.load(str(INDEX_PATH), str(DOCMAP_PATH), str(TF_PATH))
            except FileNotFoundError:
                print("Index not found. Please build the index first.")
                exit(1)

            tokens = index.tokenize(normalize(args.query))
            filtered_tokens = [token for token in tokens if token not in stop_words]
            if not filtered_tokens:
                print("No searchable tokens after stopword filtering.")
                return

            scores: Counter[str] = Counter()
            for token in filtered_tokens:
                for doc_id in index.get_documents(token):
                    scores[doc_id] += 1

            if not scores:
                print("No matching documents.")
                return

            ranked = sorted(scores.items(), key=lambda item: (-item[1], item[0]))
            for doc_id, score in ranked[:10]:
                print(f"doc_id={doc_id} score={score}")

        case "build":
            documents = load_documents()
            index.build(documents)
            index.save(str(INDEX_PATH), str(DOCMAP_PATH), str(TF_PATH) , str(DOC_LENGTHS_PATH))
            print(f"Built index for {len(documents)} documents.")

        case "tf":
            try:
                index.load(str(INDEX_PATH), str(DOCMAP_PATH), str(TF_PATH), str(DOC_LENGTHS_PATH) )
            except FileNotFoundError:
                print("Index not found. Please build the index first.")
                exit(1)

            tf = index.get_tf(args.doc_id, args.term)
            print(f"Term Frequency of '{args.term}' in document '{args.doc_id}': {tf}")
        case "idf" : 
            try : 
                index.load(str(INDEX_PATH), str(DOCMAP_PATH), str(TF_PATH), str(DOC_LENGTHS_PATH))
            except FileNotFoundError :
                print ("Index not found. Please build the index first.")
                exit(1)
            idf = index.get_idf  (args.term)
            print(f"Inverse Document Frequency of '{args.term}': {idf : .2f}")
        case "tfidf" :
            try : 
                index.load(str(INDEX_PATH), str(DOCMAP_PATH), str(TF_PATH), str(DOC_LENGTHS_PATH))
            except FileNotFoundError :
                print ("Index not found. Please build the index first.")
                exit(1)
            tf = index.get_tf(args.doc_id, args.term)
            idf = index.get_idf(args.term)
            tfidf = tf * idf
            print(f"TF-IDF score of '{args.term}' in document '{args.doc_id}': {tfidf : .2f}")
        case "bm25idf" :
            try : 
                index.load(str(INDEX_PATH), str(DOCMAP_PATH), str(TF_PATH), str(DOC_LENGTHS_PATH))
            except FileNotFoundError :
                print ("Index not found. Please build the index first.")
                exit(1)
            bm25_idf = index.get_bm25_idf(args.term)
            print(f"BM25 IDF score of '{args.term}': {bm25_idf : .2f}")
        case "bm25tf" :
            try : 
                index.load(str(INDEX_PATH), str(DOCMAP_PATH), str(TF_PATH) , str(DOC_LENGTHS_PATH))
            except FileNotFoundError :
                print ("Index not found. Please build the index first.")
                exit(1)
            bm25_tf = index.get_bm25_tf(args.doc_id, args.term, b=args.b)
            print(f"BM25 TF score of '{args.term}' in document '{args.doc_id}': {bm25_tf : .2f}")
        case "bm25search" :
            try : 
                index.load(str(INDEX_PATH), str(DOCMAP_PATH), str(TF_PATH) , str(DOC_LENGTHS_PATH))
            except FileNotFoundError :
                print ("Index not found. Please build the index first.")
                exit(1)
            movie_titles = load_movie_titles()
            results = index.bm25_search(args.query, args.limit)
            if not results:
                print("No matching documents.")
                return
            for i, (doc_id, score) in enumerate(results, 1):
                title = movie_titles.get(str(doc_id), index.get_document_title(doc_id))
                print(f"{i}. ({doc_id}) {title} - Score: {score:.2f}")
        case _:
            print("Unknown command")


if __name__ == "__main__":
    main()
