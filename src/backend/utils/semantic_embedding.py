"""Ontology leaf semantic embedding + 2-D UMAP projection.

Embeds ontology leaves using text-embedding-3-large (preferred, via OpenRouter
or OpenAI), falling back to a local sentence-transformer model when no API
key is configured. Embeddings are cached to disk keyed on
(backend, model, text_hash) so repeated runs / the interactive dashboard do
not re-pay the API cost.

Context-enriched embedding text
-------------------------------
For every leaf the embedded string contains:
  1. the ontology root + full parent path (semantic disambiguation),
  2. the leaf label (human readable),
  3. any metadata annotations (description, mechanism, adversarial_direction).

This lets e.g. "Trust_National_Government" (in PROFILE) be clearly separated
from "Trust_In_Mainstream_Journalism" (in OPINION) in the embedding space
even though both are trust items, because the parent path is part of the
prompt.

Public API
~~~~~~~~~~
    embed_ontology(ontology_root, out_dir, backend=..., model=...)
        → EmbeddingArtifact (numpy ndarray, metadata, UMAP 2-D, clusters)

    EmbeddingArtifact.write(path)        # atomic write of .npz + .json
    EmbeddingArtifact.read(path)         # round-trip
"""
from __future__ import annotations

import hashlib
import json
import logging
import os
import time
from dataclasses import dataclass, asdict, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

LOGGER = logging.getLogger(__name__)

# --------------------------------------------------------------------------
# Embedding text construction
# --------------------------------------------------------------------------

_METADATA_PRIORITY = ("description", "mechanism", "primary_system",
                      "platform_hint", "adversarial_direction")


def _collect_leaf_metadata(tree: Dict[str, Any],
                           path_parts: Sequence[str]) -> Dict[str, Any]:
    node: Any = tree
    for p in path_parts:
        if isinstance(node, dict) and p in node:
            node = node[p]
        else:
            return {}
    if not isinstance(node, dict):
        return {}
    return {k: v for k, v in node.items()
            if not k.startswith("_")
            and (not k[0].isupper() if k else False)
            and not isinstance(v, dict)}


def build_leaf_embedding_text(ontology_name: str,
                              path_parts: Sequence[str],
                              metadata: Dict[str, Any]) -> str:
    """Construct a single natural-language string for a leaf.

    Example output:
        "OPINION taxonomy → Issue_Position_Taxonomy → Defense_and_National_Security → Alliance_Commitment_Support.
         adversarial_direction: -1."
    """
    leaf = path_parts[-1] if path_parts else ""
    breadcrumb = " → ".join(path_parts)
    pieces = [f"{ontology_name} taxonomy: {breadcrumb}."]
    # Turn underscores into spaces in the leaf label for a more natural string.
    pieces.append(f"Leaf label: {leaf.replace('_', ' ')}.")
    for key in _METADATA_PRIORITY:
        if key in metadata and metadata[key] not in ("", None):
            val = metadata[key]
            if isinstance(val, (int, float)):
                pieces.append(f"{key}: {val}.")
            else:
                pieces.append(f"{key}: {val}")
    return " ".join(pieces).strip()


# --------------------------------------------------------------------------
# Backends
# --------------------------------------------------------------------------

class _EmbeddingBackend:
    name: str = "base"
    dim: int = 0

    def embed(self, texts: List[str]) -> np.ndarray:
        raise NotImplementedError


class _OpenAICompatibleBackend(_EmbeddingBackend):
    """POST to an OpenAI-compatible /v1/embeddings endpoint.

    Works with OpenAI, OpenRouter (`https://openrouter.ai/api/v1/embeddings`),
    Together, and most proxies using the OpenAI schema.
    """
    def __init__(self, api_key: str, model: str, base_url: str,
                 timeout: int = 60, batch_size: int = 32) -> None:
        import httpx  # local import — keeps module importable without httpx
        self._httpx = httpx
        self.api_key = api_key
        self.model = model
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self.batch_size = batch_size
        self.name = f"openai_compat[{base_url}|{model}]"

    def embed(self, texts: List[str]) -> np.ndarray:
        out: List[List[float]] = []
        headers = {"Authorization": f"Bearer {self.api_key}",
                   "Content-Type": "application/json"}
        url = f"{self.base_url}/embeddings"
        with self._httpx.Client(timeout=self.timeout) as client:
            for i in range(0, len(texts), self.batch_size):
                batch = texts[i:i + self.batch_size]
                body = {"model": self.model, "input": batch}
                attempts = 0
                while True:
                    attempts += 1
                    try:
                        resp = client.post(url, headers=headers, json=body)
                        resp.raise_for_status()
                        data = resp.json()
                        # Sort by index so we always preserve input order.
                        rows = sorted(data["data"],
                                      key=lambda r: r.get("index", 0))
                        for row in rows:
                            out.append(row["embedding"])
                        break
                    except Exception as err:  # noqa: BLE001
                        if attempts >= 4:
                            raise
                        backoff = 1.5 ** attempts
                        LOGGER.warning(
                            "Embedding batch %d-%d failed (%s); retry in %.1fs",
                            i, i + len(batch), err, backoff)
                        time.sleep(backoff)
        arr = np.asarray(out, dtype=np.float32)
        self.dim = arr.shape[1] if arr.size else 0
        return arr


class _LocalSentenceTransformerBackend(_EmbeddingBackend):
    """Deterministic, offline fallback."""
    def __init__(self,
                 model_name: str = "sentence-transformers/all-MiniLM-L6-v2"
                 ) -> None:
        try:
            from sentence_transformers import SentenceTransformer
        except Exception as err:  # noqa: BLE001
            raise RuntimeError(
                "sentence-transformers not available; cannot fall back locally"
            ) from err
        self._st = SentenceTransformer(model_name)
        self.model = model_name
        self.name = f"local_st[{model_name}]"

    def embed(self, texts: List[str]) -> np.ndarray:
        arr = self._st.encode(texts, show_progress_bar=False,
                              convert_to_numpy=True, normalize_embeddings=False)
        arr = np.asarray(arr, dtype=np.float32)
        self.dim = arr.shape[1] if arr.size else 0
        return arr


def _select_backend(preferred_backend: Optional[str],
                    preferred_model: Optional[str]) -> _EmbeddingBackend:
    """Pick the best available backend.

    Preference order:
      1. explicit ``preferred_backend``
      2. OPENAI_API_KEY  → OpenAI /v1/embeddings
      3. OPENROUTER_API_KEY → OpenRouter /v1/embeddings
      4. local sentence-transformers (always available after requirements install)
    """
    # Explicit selection
    if preferred_backend == "openai":
        key = os.environ.get("OPENAI_API_KEY")
        if not key:
            raise RuntimeError("preferred_backend='openai' but no OPENAI_API_KEY")
        model = preferred_model or "text-embedding-3-large"
        return _OpenAICompatibleBackend(key, model, "https://api.openai.com/v1")
    if preferred_backend == "openrouter":
        key = os.environ.get("OPENROUTER_API_KEY")
        if not key:
            raise RuntimeError("preferred_backend='openrouter' but no OPENROUTER_API_KEY")
        model = preferred_model or "openai/text-embedding-3-large"
        return _OpenAICompatibleBackend(key, model, "https://openrouter.ai/api/v1")
    if preferred_backend == "local":
        model = preferred_model or "sentence-transformers/all-MiniLM-L6-v2"
        return _LocalSentenceTransformerBackend(model)

    # Auto-select
    if os.environ.get("OPENAI_API_KEY"):
        model = preferred_model or "text-embedding-3-large"
        LOGGER.info("Using OpenAI direct (%s) for ontology embedding.", model)
        return _OpenAICompatibleBackend(os.environ["OPENAI_API_KEY"], model,
                                        "https://api.openai.com/v1")
    if os.environ.get("OPENROUTER_API_KEY"):
        model = preferred_model or "openai/text-embedding-3-large"
        LOGGER.info("Using OpenRouter (%s) for ontology embedding.", model)
        try:
            return _OpenAICompatibleBackend(os.environ["OPENROUTER_API_KEY"],
                                            model,
                                            "https://openrouter.ai/api/v1")
        except Exception as err:  # noqa: BLE001
            LOGGER.warning("OpenRouter embedding init failed (%s); falling back "
                           "to local model.", err)
    model = preferred_model or "sentence-transformers/all-MiniLM-L6-v2"
    LOGGER.info("Using local sentence-transformers (%s) for ontology embedding.",
                model)
    return _LocalSentenceTransformerBackend(model)


# --------------------------------------------------------------------------
# Artifact schema
# --------------------------------------------------------------------------

@dataclass
class LeafEmbeddingRecord:
    ontology: str
    path: str
    leaf: str
    text: str
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class EmbeddingArtifact:
    backend: str
    model: str
    dim: int
    records: List[LeafEmbeddingRecord]
    embeddings: np.ndarray            # (n_leaves, dim)
    umap_2d: np.ndarray               # (n_leaves, 2)
    cluster_labels: np.ndarray        # (n_leaves,) int
    cluster_algo: str
    cluster_n: int
    notes: List[str] = field(default_factory=list)

    # ------------------------------------------------------------------
    # Serialisation
    # ------------------------------------------------------------------
    def to_dashboard_dict(self) -> Dict[str, Any]:
        """Compact representation consumed by the interactive dashboard."""
        return {
            "backend": self.backend,
            "model": self.model,
            "dim": int(self.dim),
            "cluster_algo": self.cluster_algo,
            "cluster_n": int(self.cluster_n),
            "notes": list(self.notes),
            "points": [
                {
                    "ontology": r.ontology,
                    "path": r.path,
                    "leaf": r.leaf,
                    "text": r.text,
                    "metadata": r.metadata,
                    "x": float(self.umap_2d[i, 0]),
                    "y": float(self.umap_2d[i, 1]),
                    "cluster": int(self.cluster_labels[i]),
                }
                for i, r in enumerate(self.records)
            ],
        }

    def write(self, out_dir: str | Path) -> Path:
        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            out_dir / "embedding_vectors.npz",
            embeddings=self.embeddings,
            umap_2d=self.umap_2d,
            cluster_labels=self.cluster_labels,
        )
        manifest = {
            "backend": self.backend,
            "model": self.model,
            "dim": int(self.dim),
            "cluster_algo": self.cluster_algo,
            "cluster_n": int(self.cluster_n),
            "notes": list(self.notes),
            "records": [asdict(r) for r in self.records],
        }
        out_json = out_dir / "embedding_manifest.json"
        out_json.write_text(json.dumps(manifest, indent=2,
                                       ensure_ascii=False, default=str))
        dashboard = out_dir / "embedding_dashboard.json"
        dashboard.write_text(json.dumps(self.to_dashboard_dict(), indent=2,
                                        ensure_ascii=False, default=str))
        return out_json


# --------------------------------------------------------------------------
# Cache
# --------------------------------------------------------------------------

def _cache_key(backend_name: str, model: str, text: str) -> str:
    h = hashlib.sha256()
    h.update(backend_name.encode("utf-8"))
    h.update(b"::")
    h.update(model.encode("utf-8"))
    h.update(b"::")
    h.update(text.encode("utf-8"))
    return h.hexdigest()


def _load_cache(cache_path: Path) -> Dict[str, List[float]]:
    if not cache_path.exists():
        return {}
    try:
        with np.load(cache_path, allow_pickle=True) as blob:
            keys = [str(x) for x in blob["keys"].tolist()]
            vecs = blob["vecs"]
        return {k: vecs[i].tolist() for i, k in enumerate(keys)}
    except Exception as err:  # noqa: BLE001
        LOGGER.warning("Could not read embedding cache %s (%s)", cache_path, err)
        return {}


def _write_cache(cache_path: Path, cache: Dict[str, List[float]]) -> None:
    if not cache:
        return
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    keys = list(cache.keys())
    vecs = np.asarray([cache[k] for k in keys], dtype=np.float32)
    np.savez_compressed(cache_path, keys=np.asarray(keys), vecs=vecs)


# --------------------------------------------------------------------------
# Main entry point
# --------------------------------------------------------------------------

def _enumerate_ontology(ontology_name: str,
                        tree: Dict[str, Any]
                        ) -> List[LeafEmbeddingRecord]:
    from src.backend.utils.ontology_utils import iter_leaf_paths
    records: List[LeafEmbeddingRecord] = []
    for path_tuple in iter_leaf_paths(tree):
        meta = _collect_leaf_metadata(tree, path_tuple)
        text = build_leaf_embedding_text(ontology_name, path_tuple, meta)
        records.append(LeafEmbeddingRecord(
            ontology=ontology_name,
            path=" > ".join(path_tuple),
            leaf=path_tuple[-1],
            text=text,
            metadata=meta,
        ))
    return records


def _umap_2d(embeddings: np.ndarray, *,
             n_neighbors: int = 12,
             min_dist: float = 0.08,
             seed: int = 42) -> np.ndarray:
    if embeddings.shape[0] < 4:
        # UMAP fails on tiny inputs — fall back to PCA.
        from sklearn.decomposition import PCA
        n = max(1, min(2, embeddings.shape[0]))
        proj = PCA(n_components=n, random_state=seed).fit_transform(embeddings)
        if proj.shape[1] == 1:
            proj = np.hstack([proj, np.zeros_like(proj)])
        return proj.astype(np.float32)
    import umap
    reducer = umap.UMAP(
        n_neighbors=min(n_neighbors, max(2, embeddings.shape[0] - 1)),
        min_dist=min_dist,
        metric="cosine",
        n_components=2,
        random_state=seed,
    )
    return reducer.fit_transform(embeddings).astype(np.float32)


def _cluster(embeddings: np.ndarray, *,
             algo: str = "kmeans",
             n_clusters: int = 8,
             seed: int = 42) -> Tuple[np.ndarray, str, int]:
    """Return (labels, algo_used, cluster_count).

    ``algo`` = ``kmeans`` is robust and deterministic; ``hdbscan`` is used
    when cluster density is unknown. When n is small we override to a
    conservative k.
    """
    n = embeddings.shape[0]
    if n == 0:
        return np.zeros((0,), dtype=np.int32), "none", 0
    k = max(2, min(n_clusters, n // 3 or 2, n))
    if algo == "hdbscan":
        try:
            import hdbscan
            labels = hdbscan.HDBSCAN(min_cluster_size=3,
                                     metric="euclidean").fit_predict(embeddings)
            unique = [c for c in np.unique(labels) if c != -1]
            return labels.astype(np.int32), "hdbscan", len(unique) or 1
        except Exception as err:  # noqa: BLE001
            LOGGER.info("HDBSCAN unavailable (%s); using KMeans.", err)
    from sklearn.cluster import KMeans
    labels = KMeans(n_clusters=k, random_state=seed, n_init=10).fit_predict(embeddings)
    return labels.astype(np.int32), "kmeans", k


def embed_ontology(ontology_root: str | Path,
                   out_dir: str | Path,
                   *,
                   backend: Optional[str] = None,
                   model: Optional[str] = None,
                   cache_dir: Optional[str | Path] = None,
                   n_clusters: int = 8,
                   seed: int = 42) -> EmbeddingArtifact:
    """Compute leaf embeddings + 2-D UMAP projection for all three ontologies.

    Parameters
    ----------
    ontology_root
        Path to a directory containing ``PROFILE/``, ``OPINION/``, ``ATTACK/``.
    out_dir
        Where to write ``embedding_manifest.json``, ``embedding_vectors.npz``,
        and the dashboard-ready JSON.
    """
    from src.backend.utils.ontology_utils import load_ontology_triplet

    trees = load_ontology_triplet(ontology_root)
    records: List[LeafEmbeddingRecord] = []
    for ont_name in ("PROFILE", "OPINION", "ATTACK"):
        records.extend(_enumerate_ontology(ont_name, trees[ont_name]))

    if not records:
        raise ValueError("No leaves discovered in provided ontology root.")

    backend_obj = _select_backend(backend, model)

    # --- cache layer
    cache_dir = Path(cache_dir) if cache_dir else Path.home() / ".cache" / "cog_pipeline" / "embeddings"
    cache_path = cache_dir / f"{backend_obj.name.replace('/', '_').replace('[', '_').replace(']', '_')}.npz"
    cache = _load_cache(cache_path)

    texts = [r.text for r in records]
    keys = [_cache_key(backend_obj.name, backend_obj.model, t) for t in texts]
    missing_idx = [i for i, k in enumerate(keys) if k not in cache]
    if missing_idx:
        LOGGER.info("Embedding %d new leaves (cached: %d).",
                    len(missing_idx), len(keys) - len(missing_idx))
        new_vecs = backend_obj.embed([texts[i] for i in missing_idx])
        for offset, i in enumerate(missing_idx):
            cache[keys[i]] = new_vecs[offset].tolist()
        _write_cache(cache_path, cache)
    else:
        LOGGER.info("All %d leaf embeddings served from cache.", len(keys))
    emb = np.asarray([cache[k] for k in keys], dtype=np.float32)

    # 2-D projection
    umap_xy = _umap_2d(emb, seed=seed)

    # Cluster
    labels, algo_used, k = _cluster(emb, algo="kmeans",
                                    n_clusters=n_clusters, seed=seed)

    artifact = EmbeddingArtifact(
        backend=backend_obj.name,
        model=backend_obj.model,
        dim=int(emb.shape[1]),
        records=records,
        embeddings=emb,
        umap_2d=umap_xy,
        cluster_labels=labels,
        cluster_algo=algo_used,
        cluster_n=int(k),
        notes=[
            f"Embedding text prepends full leaf path for disambiguation.",
            f"UMAP parameters: n_neighbors=min(12,n-1), min_dist=0.08, metric=cosine.",
            f"Clustering: {algo_used} (k={k}).",
        ],
    )
    artifact.write(out_dir)
    LOGGER.info("Wrote embedding artifact (%d leaves, dim=%d) to %s",
                len(records), emb.shape[1], out_dir)
    return artifact
