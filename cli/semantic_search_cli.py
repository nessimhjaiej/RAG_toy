#!/usr/bin/env python3

import argparse
import json
import re
from pathlib import Path

from lib.semantic_search import SemanticSearch, verify_model
from lib.semantic_search import embed_text
from lib.semantic_search import verify_embeddings
from lib.semantic_search import embed_query_text

PROJECT_ROOT = Path(__file__).resolve().parents[1]
MOVIES_PATH = PROJECT_ROOT / "data" / "movies.json"


def load_documents():
    with MOVIES_PATH.open("r", encoding="utf-8") as f:
        payload = json.load(f)
    return payload.get("movies", [])


def main():
    parser = argparse.ArgumentParser(description="Semantic Search CLI")
    parser.add_argument("command", help="Command to run")
    parser.add_argument("--limit", type=int, default=5, help="Number of results to return (default: 5)")
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=200,
        help="Number of words per chunk for the 'chunk' command (default: 200)",
    )
    parser.add_argument("--overlap" , type=int, default=50, help="Number of overlapping words between chunks for the 'chunk' command (default: 50)")
    parser.add_argument("--max-chunk-size", type=int, default=4, help="Maximum number of words per chunk for the 'semantic_chunk' command (default: 200)")
    parser.add_argument("--overlap-size", type=int, default=0, help="Number of overlapping words between chunks for the 'semantic_chunk' command (default: 50)")
    args = parser.parse_args()


    if args.chunk_size <= 0:
        parser.error("--chunk-size must be a positive integer")
    if args.max_chunk_size <= 0:
        parser.error("--max-chunk-size must be a positive integer")
    if args.overlap_size < 0:
        parser.error("--overlap-size cannot be negative")
    if args.overlap_size >= args.max_chunk_size:
        parser.error("--overlap-size must be smaller than --max-chunk-size")

    match args.command:
        
        case "verify" : 
            verify_model()
        case "embed_text"  :
            text = input("Enter text to embed: ")
            if len(text.strip()) == 0 : 
                print("Text cannot be empty.")
                return
            if len(text) > 512 : 
                print("Text exceeds maximum length of 512 characters.")
                return
            #only accepting one string 
            embed_text(text)
        case "verify_embeddings" :
            verify_embeddings()
        case "embed_query" :
            query = input("Enter query text to embed: ")
            if len(query.strip()) == 0 : 
                print("Query text cannot be empty.")
                return
            embed_query_text(query)
        case "search" : 
            query = input("Enter search query: ")
            if len(query.strip()) == 0 : 
                print("Query text cannot be empty.")
                return
            search = SemanticSearch()
            documents = load_documents()
            search.load_or_create_embeddings(documents)
            results = search.search(query, args.limit)
            for result in results:
                print(f"Title: {result['title']}, Score: {result['score']:.2f}")
        case "chunk" : 
            input_text = input("Enter text to chunk: ")
            chunk_size = args.chunk_size
            overlap_size = args.overlap
            input_text = input_text.split()
            n = chunk_size
            chunks = [input_text[i:i + n] for i in range(0, len(input_text), n)]
            if overlap_size > 0 and len(chunks) > 1:
                for i in range(1, len(chunks)):
                    overlap = input_text[i * n - overlap_size:i * n]
                    chunks[i] = overlap + chunks[i]
                    print (f"Chunk {i}: {' '.join(chunks[i])}")
            else : 
                for idx, chunk in enumerate(chunks):
                    print(f"Chunk {idx + 1}: {' '.join(chunk)}")
        case "semantic_chunk" :
            input_text = input("Enter text to chunk: ")
            max_chunk_size = args.max_chunk_size
            overlap_size = args.overlap_size
            # Split into sentences on '.' (also works when no space follows '.').
            input_text = [s.strip() for s in re.split(r"(?<=\.)\s*", input_text.strip()) if s.strip()]
            chunks = []
            for i in range(0, len(input_text), max_chunk_size):
                start = max(0, i - overlap_size)
                end = i + max_chunk_size
                chunk = input_text[start:end]
                chunks.append(chunk)
            for idx, chunk in enumerate(chunks):
                print(f"Chunk {idx + 1}: {' '.join(chunk)}")
        case _:
            parser.print_help()


if __name__ == "__main__":
    main()
