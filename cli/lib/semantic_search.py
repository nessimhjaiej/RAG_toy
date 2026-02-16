from pathlib import Path
import json
import re 
import numpy as np
from sentence_transformers import SentenceTransformer

PROJECT_ROOT = Path(__file__).resolve().parents[2]
CACHE_FILE = PROJECT_ROOT / "cache" / "movie_embeddings.npy"
MOVIES_FILE = PROJECT_ROOT / "data" / "movies.json"


class SemanticSearch:
    def __init__(self, model_name="all-MiniLM-L6-v2", embeddings=None, documents=None, document_map=None):
        self.model_name = model_name
        self.model = None
        self.embeddings = embeddings if embeddings is not None else []
        self.documents = documents if documents is not None else []
        self.document_map = document_map if document_map is not None else {}

    def _get_model(self):
        if self.model is None:
            self.model = SentenceTransformer(self.model_name)
        return self.model

    def _normalize_documents(self, documents):
        if isinstance(documents, dict):
            if "movies" in documents and isinstance(documents["movies"], list):
                return documents["movies"]
            raise ValueError("Document payload dict must contain a 'movies' list.")
        if isinstance(documents, list):
            return documents
        raise ValueError("Documents must be a list or a payload dict containing 'movies'.")

    def _document_to_text(self, doc):
        if isinstance(doc, dict):
            title = str(doc.get("title", "")).strip()
            description = str(doc.get("description", "")).strip()
            if title and description:
                return f"{title}: {description}"
            if title:
                return title
            if description:
                return description
            return json.dumps(doc, ensure_ascii=False)
        return str(doc).strip()

    def generate_embedding(self, text):
        if text is None or text.strip() == "":
            raise ValueError("Input text cannot be empty.")
        text = [text]  # Convert to list for batch processing
        embedding = self._get_model().encode(text)
        return embedding[0]  # Return the first (and only) embedding
    def build_embeddings(self, documents):
        normalized_documents = self._normalize_documents(documents)
        if not normalized_documents:
            raise ValueError("Document list cannot be empty.")
        self.documents = normalized_documents

        self.document_map = {id: doc for id, doc in enumerate(self.documents)}
        document_strings = [self._document_to_text(doc) for doc in self.documents]
        self.embeddings = self._get_model().encode(document_strings, show_progress_bar=True)
        CACHE_FILE.parent.mkdir(parents=True, exist_ok=True)
        np.save(CACHE_FILE, self.embeddings)
        return self.embeddings

    def load_or_create_embeddings(self, documents):
        self.documents = self._normalize_documents(documents)
        self.document_map = {id: doc for id, doc in enumerate(self.documents)}
        if CACHE_FILE.exists():
            self.embeddings = np.load(CACHE_FILE)
        if len(self.embeddings) != len(self.documents):
            print("Embedding count does not match document count. Rebuilding embeddings.")
            return self.build_embeddings(self.documents)
        return self.embeddings
    def search(self, query, limit): 
        if len(self.embeddings) == 0 or len(self.documents) == 0:
            raise ValueError("Embeddings and documents must be loaded before searching.")
        query_embedding = self.generate_embedding(query)
        similarities = [cosine_similarity(query_embedding, doc_emb) for doc_emb in self.embeddings]
        top_indices = np.argsort(similarities)[::-1][:limit]
        #return should contain the title , score and the description 
        results = []
        for idx in top_indices:
            doc = self.document_map[idx]
            title = doc.get("title", "No Title")
            description = doc.get("description", "No Description")
            score = similarities[idx]
            results.append({"title": title, "description": description, "score": score})
        return results
        


def verify_model():
    try:
        model = SentenceTransformer("all-MiniLM-L6-v2")
        print(f"Model loaded successfully : {model.__class__.__name__}")
        print(f"Max sequence length: {model.max_seq_length}")
    except Exception as e:
        print(f"Error loading model: {e}")


def embed_text(text):
    try:
        search = SemanticSearch()
        embedding = search.generate_embedding(text)
        print(f"Text: {text}")
        print(f"First 3 dimensions: {embedding[:3]}")
        print(f"Dimensions: {embedding.shape[0]}")
    except Exception as e:
        print(f"Error generating embedding: {e}")
        print("Ensure internet access is available at first run so the model can be downloaded.")


def verify_embeddings():
    try:
        search = SemanticSearch()
        with MOVIES_FILE.open("r", encoding="utf-8") as f:
            payload = json.load(f)
        documents = payload.get("movies", payload)
        search.load_or_create_embeddings(documents)
        print(f"Number of docs:   {len(documents)}")
        print(f"Embeddings shape: {search.embeddings.shape[0]} vectors in {search.embeddings.shape[1]} dimensions")
    except Exception as e:
        print(f"Error verifying embeddings: {e}")
        print("If this is the first run, allow internet access to download the model, then retry.")

def embed_query_text(query)  : 
    if query is None or query.strip() == "":
        raise ValueError("Query text cannot be empty.")
    search = SemanticSearch()
    embedding = search.generate_embedding(query)
    print(f"Query: {query}")
    print(f"First 5 dimensions: {embedding[:5]}")
    print(f"Shape: {embedding.shape}")

def cosine_similarity(vec1, vec2):
    dot_product = np.dot(vec1, vec2)
    norm1 = np.linalg.norm(vec1)
    norm2 = np.linalg.norm(vec2)

    if norm1 == 0 or norm2 == 0:
        return 0.0

    return dot_product / (norm1 * norm2)


class ChunkedSemanticSearch(SemanticSearch):
    def __init__(self, model_name = "all-MiniLM-L6-v2") -> None:
        super().__init__(model_name)
        self.chunk_embeddings = None
        self.chunk_metadata = None
    def build_chunk_embeddings(self, documents) : 
        self.documents= documents
        self.document_map = {id: doc for id, doc in enumerate(self.documents)}
        chunks = []
        #holding metadata for each chunk (list of dictionnaries)
        metadata = []
        for doc_id, doc in self.document_map.items() :
            if doc is None : 
                continue
            text = self._document_to_text(doc)
            #split into sentences 4 sentences per chunk with 1 overlap
            sentences = [s.strip() for s in re.split(r"(?<=[.!?])\s+", text) if s.strip()]
            for i in range(0, len(sentences), 4) :
                chunk = sentences[i:i+4]
                chunks.append(" ".join(chunk))
                metadata.append({"doc_id": doc_id, "chunk_index": i//4, "total_chunks": (len(sentences) + 3) // 4})
        self.chunk_metadata = metadata
        self.chunk_embeddings = self._get_model().encode(chunks, show_progress_bar=True)
        np.save(CACHE_FILE.with_name("chunk_embeddings.npy"), self.chunk_embeddings)
        with CACHE_FILE.with_name("chunk_metadata.json").open("w", encoding="utf-8") as f:
            json.dump({"chunks": metadata, "total_chunks": len(chunks)},    f, indent=2)
        return self.chunk_embeddings
    def load_or_create_chunk_embeddings(self, documents: list[dict]) -> np.ndarray  : 
        self.documents = documents
        self.document_map = {id: doc for id, doc in enumerate(self.documents)}
        chunk_cache_file = CACHE_FILE.with_name("chunk_embeddings.npy")
        metadata_cache_file = CACHE_FILE.with_name("chunk_metadata.json")
        if chunk_cache_file.exists() and metadata_cache_file.exists():
            self.chunk_embeddings = np.load(chunk_cache_file)
            with metadata_cache_file.open("r", encoding="utf-8") as f:
                metadata_payload = json.load(f)
                self.chunk_metadata = metadata_payload.get("chunks", [])
            if len(self.chunk_embeddings) != len(self.chunk_metadata):
                print("Chunk embedding count does not match metadata count. Rebuilding chunk embeddings.")
                return self.build_chunk_embeddings(documents)
            return self.chunk_embeddings
        return self.build_chunk_embeddings(documents)
        