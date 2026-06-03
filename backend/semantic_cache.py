"""
Semantic cache for RAG queries.

On each /ask request:
  1. Embed the question (same model already loaded for FAISS).
  2. Compare against every cached question embedding via cosine similarity.
  3. If best similarity >= CACHE_SIMILARITY_THRESHOLD  →  return cached answer (no LLM call).
  4. Otherwise run the full pipeline, then store the result.

Persistence: vector_store/semantic_cache.pkl  (auto-created).
Eviction   : TTL (CACHE_TTL_DAYS) + LRU cap (CACHE_MAX_SIZE).
"""

import os
import re
import time
import pickle

import numpy as np

from backend.config import (
    VECTOR_STORE_DIR,
    CACHE_CANDIDATE_FLOOR,
    CACHE_TTL_DAYS,
    CACHE_MAX_SIZE,
)

CACHE_PATH = os.path.join(VECTOR_STORE_DIR, "semantic_cache.pkl")

# Common English words that carry no topical meaning
_STOPWORDS = frozenset({
    "a", "an", "the", "is", "are", "was", "were", "be", "been", "being",
    "have", "has", "had", "do", "does", "did", "will", "would", "could",
    "should", "may", "might", "shall", "can", "need", "dare", "ought",
    "what", "which", "who", "whom", "whose", "where", "when", "why", "how",
    "he", "she", "it", "they", "we", "you", "i", "his", "her", "its",
    "their", "our", "your", "my", "this", "that", "these", "those",
    "of", "in", "on", "at", "to", "for", "by", "from", "with", "about",
    "into", "through", "during", "before", "after", "above", "below",
    "and", "or", "but", "if", "then", "because", "as", "until", "while",
    "tell", "me", "give", "show", "explain", "describe", "please", "list",
    "get", "find", "any", "all", "some", "there", "here", "just", "than",
    "too", "very", "so", "yet", "both", "each", "more", "most", "up",
    "out", "off", "no", "not", "only", "same", "also", "than", "then",
    # Resume-context structural words (refer to the person, not the topic)
    "candidate", "applicant", "person", "employee", "worker", "individual",
    "resume", "cv", "profile", "document",
    # Weak qualifier words that don't differentiate topic
    "level", "type", "kind", "current", "latest", "recent", "last",
    "overall", "general", "main", "key", "top", "best",
})

# ---------------------------------------------------------------------------
# Synonym map — normalise vocabulary variants to a shared canonical token
# so "studied" and "education level" both resolve to "education", etc.
# ---------------------------------------------------------------------------

_SYNONYMS: dict[str, str] = {
    # Education
    "education": "education", "studied": "education", "study": "education",
    "studying": "education", "degree": "education", "qualification": "education",
    "qualifications": "education", "academic": "education", "academics": "education",
    "graduate": "education", "graduated": "education", "graduation": "education",
    "school": "education", "college": "education", "university": "education",
    "bachelor": "education", "masters": "education", "master": "education",
    "phd": "education", "diploma": "education", "course": "education",

    # Skills
    "skills": "skills", "skill": "skills", "expertise": "skills",
    "competency": "skills", "competencies": "skills", "proficiency": "skills",
    "proficient": "skills", "technology": "skills", "technologies": "skills",
    "tools": "skills", "programming": "skills", "languages": "skills",
    "tech": "skills", "technical": "skills", "abilities": "skills",
    "ability": "skills", "knowledge": "skills", "stack": "skills",

    # Work experience
    "experience": "experience", "work": "experience", "worked": "experience",
    "working": "experience", "job": "experience", "jobs": "experience",
    "role": "experience", "roles": "experience", "position": "experience",
    "positions": "experience", "career": "experience", "employment": "experience",
    "employed": "experience", "company": "experience", "companies": "experience",
    "employer": "experience", "employers": "experience", "industry": "experience",

    # Name / identity
    "name": "name", "named": "name", "called": "name", "known": "name",
    "surname": "name", "firstname": "name", "fullname": "name",
    "identity": "name", "identify": "name",

    # Achievements / awards
    "achievement": "achievement", "achievements": "achievement",
    "accomplishment": "achievement", "accomplishments": "achievement",
    "award": "achievement", "awards": "achievement", "recognition": "achievement",
    "recognized": "achievement", "honored": "achievement", "honour": "achievement",
    "performance": "achievement", "star": "achievement",

    # Certifications
    "certification": "certification", "certifications": "certification",
    "certified": "certification", "certificate": "certification",
    "certificates": "certification", "license": "certification",
    "licensed": "certification", "credential": "certification",

    # Location
    "location": "location", "city": "location", "address": "location",
    "based": "location", "located": "location", "place": "location",
    "lives": "location", "residing": "location", "resident": "location",
    "hometown": "location", "country": "location", "state": "location",

    # Contact
    "contact": "contact", "email": "contact", "phone": "contact",
    "mobile": "contact", "number": "contact", "linkedin": "contact",
    "reach": "contact", "reachable": "contact",

    # Summary / profile
    "summary": "summary", "profile": "summary", "overview": "summary",
    "introduction": "summary", "background": "summary", "about": "summary",
    "objective": "summary", "goal": "summary", "goals": "summary",
}


def _keyword_score(a: str, b: str) -> float:
    """
    Token containment score after:
      1. Punctuation stripping + lowercase
      2. Stopword removal
      3. Synonym normalisation  (e.g. "studied" → "education", "worked" → "experience")

    Returns max(|A∩B|/|A|, |A∩B|/|B|) so that when a short query's key terms
    are fully contained in a longer query the score is 1.0.

    Examples
    --------
    "name?"              vs "what was the name of the candidate?"  → 1.0  (HYBRID HIT)
    "what he studied?"   vs "what was the education level?"        → 1.0  (HYBRID HIT)
    "name?"              vs "what are his skills?"                 → 0.0  (MISS)
    "education level"    vs "what was the name?"                   → 0.0  (MISS)
    """
    def _tokens(text: str) -> set:
        words = re.sub(r"[^a-z0-9 ]", " ", text.lower()).split()
        result = set()
        for w in words:
            if w in _STOPWORDS or len(w) <= 1:
                continue
            # Normalise to canonical topic token if a synonym exists
            result.add(_SYNONYMS.get(w, w))
        return result

    ta, tb = _tokens(a), _tokens(b)
    if not ta or not tb:
        return 0.0
    common = ta & tb
    if not common:
        return 0.0
    return max(len(common) / len(ta), len(common) / len(tb))


class SemanticCache:
    """
    In-memory cache backed by a pickle file.

    Each entry stores:
        embedding  – normalised numpy vector of the question
        question   – original question text (for logging / debug)
        answer     – LLM answer string
        sources    – list of source dicts
        ts         – unix timestamp of last access (used for TTL + LRU ordering)
    """

    def __init__(self):
        self._entries: list[dict] = []
        self._load()

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def _load(self):
        if os.path.exists(CACHE_PATH):
            try:
                with open(CACHE_PATH, "rb") as f:
                    self._entries = pickle.load(f)
                cutoff = time.time() - CACHE_TTL_DAYS * 86_400
                before = len(self._entries)
                self._entries = [e for e in self._entries if e["ts"] >= cutoff]
                evicted = before - len(self._entries)
                if evicted:
                    print(f"[CACHE] Evicted {evicted} expired entries on load.")
            except Exception as exc:
                print(f"[CACHE] Failed to load cache ({exc}), starting fresh.")
                self._entries = []

    def _save(self):
        os.makedirs(VECTOR_STORE_DIR, exist_ok=True)
        with open(CACHE_PATH, "wb") as f:
            pickle.dump(self._entries, f)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def get(self, embedding: np.ndarray, question: str) -> dict | None:
        """
        Return a cache candidate if cosine similarity >= CACHE_CANDIDATE_FLOOR.

        The caller (rag_pipeline) is responsible for asking the LLM whether the
        candidate actually answers the new question before using it.  The returned
        dict contains '_needs_validation': True as an internal signal.

        Returns None when no candidate meets the floor (definite miss).
        """
        if not self._entries:
            return None

        matrix = np.stack([e["embedding"] for e in self._entries])  # (N, D)
        sims = matrix @ embedding                                     # cosine (already normalised)

        best_idx = int(np.argmax(sims))
        best_sim = float(sims[best_idx])
        entry = self._entries[best_idx]

        if best_sim >= CACHE_CANDIDATE_FLOOR:
            self._entries.pop(best_idx)
            entry["ts"] = time.time()
            self._entries.insert(0, entry)          # promote to MRU position
            print(f"[CACHE CANDIDATE] sim={best_sim:.4f}  q={entry['question']!r}")
            return {
                "answer": entry["answer"],
                "sources": entry["sources"],
                "cached": True,
                "cache_similarity": round(best_sim, 4),
                "cached_question": entry["question"],
                "_needs_validation": True,
            }

        print(f"[CACHE MISS] sim={best_sim:.4f}")
        return None

    def set(
        self,
        embedding: np.ndarray,
        question: str,
        answer: str,
        sources: list,
    ) -> None:
        """Store a new question-answer pair in the cache."""
        # LRU eviction when at capacity
        while len(self._entries) >= CACHE_MAX_SIZE:
            dropped = self._entries.pop()
            print(f"[CACHE] LRU evicted: {dropped['question']!r}")

        self._entries.insert(0, {
            "embedding": embedding,
            "question": question,
            "answer": answer,
            "sources": sources,
            "ts": time.time(),
        })
        self._save()
        print(f"[CACHE STORE] size={len(self._entries)}  q={question!r}")

    def clear(self) -> int:
        """Wipe all entries. Returns how many were removed."""
        count = len(self._entries)
        self._entries = []
        if os.path.exists(CACHE_PATH):
            os.remove(CACHE_PATH)
        print(f"[CACHE] Cleared {count} entries.")
        return count

    @property
    def size(self) -> int:
        return len(self._entries)

    def stats(self) -> dict:
        return {
            "size": self.size,
            "max_size": CACHE_MAX_SIZE,
            "ttl_days": CACHE_TTL_DAYS,
            "candidate_floor": CACHE_CANDIDATE_FLOOR,
            "entries": [
                {
                    "question": e["question"],
                    "cached_at": time.strftime(
                        "%Y-%m-%d %H:%M:%S", time.localtime(e["ts"])
                    ),
                }
                for e in self._entries
            ],
        }


# ---------------------------------------------------------------------------
# Singleton accessor
# ---------------------------------------------------------------------------

_cache_instance: SemanticCache | None = None


def get_cache() -> SemanticCache:
    global _cache_instance
    if _cache_instance is None:
        _cache_instance = SemanticCache()
    return _cache_instance
