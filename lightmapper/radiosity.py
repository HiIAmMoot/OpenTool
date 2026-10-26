"""Convert incident light samples to stock vectors and share seam query points.

Coherence keeps the directional fraction, not just its direction: the unit
direction times min(|V| / E, 1). A query with no light, or a direction sum
under 1e-4, gets a zero vector as in tool, which has no normal fallback (the
normals tool's tags show are ambient-dominated vertices). The scaled mode
keeps its normal fallback.
"""

import numpy as np


def finalize(V: np.ndarray, E: np.ndarray, N: np.ndarray,
             mode: str, scale: float) -> np.ndarray:
    V = np.asarray(V, dtype=np.float64)
    E = np.asarray(E, dtype=np.float64)
    with np.errstate(divide='ignore', invalid='ignore'):
        if mode == 'coherence':
            length = np.linalg.norm(V, axis=1)
            result = V / length[:, None] * np.minimum(length / E, 1)[:, None]
            # (a negative, NaN or zero energy compares false here and so is zeroed)
            result[~((E > 0) & (length >= 1e-4) & np.isfinite(V).all(axis=1))] = 0
            return np.clip(result, -1, 1).astype(np.float32)
        elif mode == 'scaled':
            result = V * scale
            # bound components before squaring so large finite directions survive
            result /= np.maximum(np.max(np.abs(result), axis=1), 1)[:, None]
            lengths = np.linalg.norm(result, axis=1)
            result /= np.maximum(lengths, 1)[:, None]
        else:
            raise ValueError(f'unknown radiosity mode: {mode}')
        unlit = (E <= 1e-6) | (np.linalg.norm(result, axis=1) < 1e-4)
    fallback = unlit | np.isnan(result).any(axis=1)
    result[fallback] = np.asarray(N)[fallback]
    return np.clip(result, -1, 1).astype(np.float32)


def query_points(env: np.ndarray, tris: np.ndarray, tri_class: np.ndarray
                 ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Share exact-byte queries; queries touched by no triangle get class -1."""
    indices = {}
    first = []
    votes = []
    inverse = np.empty(len(env), dtype=np.intp)
    for i, (position, normal) in enumerate(zip(env['position'], env['normal'])):
        # byte keys retain signed zero and avoid float equality merging distinct normals
        key = position.tobytes() + normal.tobytes()
        query = indices.get(key)
        if query is None:
            query = len(first)
            indices[key] = query
            first.append(i)
            votes.append({})
        inverse[i] = query
    for triangle, cls in zip(inverse[tris], tri_class):
        # several corners or seam copies in one triangle still contribute only one vote
        for query in set(triangle):
            counts = votes[query]
            cls = int(cls)
            counts[cls] = counts.get(cls, 0) + 1
    query_class = np.array([min(counts, key=lambda cls: (-counts[cls], cls), default=-1)
                            for counts in votes], dtype=np.int16)
    first = np.asarray(first, dtype=np.intp)
    return env['position'][first], env['normal'][first], query_class, inverse
