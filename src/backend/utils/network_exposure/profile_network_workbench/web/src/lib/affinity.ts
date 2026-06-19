import type {
  AffinityFormulaWeights,
  ProfileLayoutAffinity,
  ProfileNetworkEdge,
  ProfileNetworkResponse
} from "../api/types";

export const DEFAULT_AFFINITY_WEIGHTS: AffinityFormulaWeights = {
  personality_similarity: 0.6,
  ontology_leaf_overlap: 0.2,
  age_context_similarity: 0.12,
  categorical_similarity: 0.08
};

const WEIGHT_KEYS: Array<keyof AffinityFormulaWeights> = [
  "personality_similarity",
  "ontology_leaf_overlap",
  "age_context_similarity",
  "categorical_similarity"
];

function clamp01(value: number) {
  if (!Number.isFinite(value)) return 0;
  return Math.max(0, Math.min(1, value));
}

function round6(value: number) {
  return Math.round(value * 1_000_000) / 1_000_000;
}

function percentile(values: number[], percentileValue: number) {
  if (!values.length) return 0;
  const ordered = values.slice().sort((left, right) => left - right);
  if (ordered.length === 1) return ordered[0];
  const position = (ordered.length - 1) * percentileValue;
  const lower = Math.floor(position);
  const upper = Math.ceil(position);
  if (lower === upper) return ordered[lower];
  const fraction = position - lower;
  return ordered[lower] * (1 - fraction) + ordered[upper] * fraction;
}

function normalizeValue(value: number, low: number, high: number) {
  if (high <= low) return 1;
  return clamp01((value - low) / (high - low));
}

export function normalizeAffinityWeights(
  weights: AffinityFormulaWeights,
  fallback: AffinityFormulaWeights = DEFAULT_AFFINITY_WEIGHTS
): AffinityFormulaWeights {
  const cleaned = Object.fromEntries(
    WEIGHT_KEYS.map((key) => [key, Math.max(0, Number(weights[key]) || 0)])
  ) as unknown as AffinityFormulaWeights;
  const total = WEIGHT_KEYS.reduce((sum, key) => sum + cleaned[key], 0);
  if (total <= 0) return { ...fallback };
  return Object.fromEntries(WEIGHT_KEYS.map((key) => [key, cleaned[key] / total])) as unknown as AffinityFormulaWeights;
}

export function affinityFromComponents(pair: ProfileLayoutAffinity, weights: AffinityFormulaWeights) {
  const normalized = normalizeAffinityWeights(weights);
  return clamp01(
    pair.components.personality_similarity * normalized.personality_similarity +
      pair.components.ontology_leaf_overlap * normalized.ontology_leaf_overlap +
      pair.components.age_context_similarity * normalized.age_context_similarity +
      pair.components.categorical_similarity * normalized.categorical_similarity
  );
}

function displayEdges(pairs: ProfileLayoutAffinity[], edgeLimitPerNode: number) {
  const byNode = new Map<string, ProfileLayoutAffinity[]>();
  pairs.forEach((pair) => {
    byNode.set(pair.source, [...(byNode.get(pair.source) ?? []), pair]);
    byNode.set(pair.target, [...(byNode.get(pair.target) ?? []), pair]);
  });
  const selected = new Map<string, ProfileLayoutAffinity>();
  byNode.forEach((nodePairs) => {
    nodePairs
      .slice()
      .sort((left, right) => right.affinity - left.affinity)
      .slice(0, edgeLimitPerNode)
      .forEach((pair) => {
        const key = pair.source < pair.target ? `${pair.source}::${pair.target}` : `${pair.target}::${pair.source}`;
        selected.set(key, pair);
      });
  });
  return Array.from(selected.values()).sort((left, right) => `${left.source}-${left.target}`.localeCompare(`${right.source}-${right.target}`));
}

function centrality(profileIds: string[], pairs: ProfileLayoutAffinity[]) {
  const raw = new Map(profileIds.map((id) => [id, [] as number[]]));
  pairs.forEach((pair) => {
    raw.get(pair.source)?.push(pair.affinity);
    raw.get(pair.target)?.push(pair.affinity);
  });
  const means = new Map<string, number>();
  raw.forEach((values, id) => {
    means.set(id, values.length ? values.reduce((sum, value) => sum + value, 0) / values.length : 0);
  });
  const values = Array.from(means.values());
  const low = Math.min(...values);
  const high = Math.max(...values);
  return new Map(Array.from(means.entries()).map(([id, value]) => [id, round6(normalizeValue(value, low, high))]));
}

function clusterIds(profileIds: string[], pairs: ProfileLayoutAffinity[]) {
  const parent = new Map(profileIds.map((id) => [id, id]));
  const threshold = percentile(
    pairs.map((pair) => pair.affinity),
    0.9
  );
  const find = (id: string): string => {
    const current = parent.get(id) ?? id;
    if (current === id) return id;
    const root = find(current);
    parent.set(id, root);
    return root;
  };
  const union = (left: string, right: string) => {
    const leftRoot = find(left);
    const rightRoot = find(right);
    if (leftRoot !== rightRoot) parent.set(rightRoot, leftRoot);
  };
  pairs.forEach((pair) => {
    if (pair.affinity >= threshold) union(pair.source, pair.target);
  });
  const components = new Map<string, string[]>();
  profileIds.forEach((id) => {
    const root = find(id);
    components.set(root, [...(components.get(root) ?? []), id]);
  });
  const ordered = Array.from(components.values()).map((ids) => ids.slice().sort()).sort((left, right) => right.length - left.length || left[0].localeCompare(right[0]));
  const byProfile = new Map<string, string>();
  ordered.forEach((ids, index) => {
    const clusterId = `cluster_${String(index + 1).padStart(2, "0")}`;
    ids.forEach((id) => byProfile.set(id, clusterId));
  });
  return byProfile;
}

export function buildWeightedNetwork(
  network: ProfileNetworkResponse,
  weights: AffinityFormulaWeights,
  edgeLimitPerNode: number
): ProfileNetworkResponse {
  const normalizedWeights = normalizeAffinityWeights(weights, network.affinity_formula.default_weights);
  const weightedPairs = network.layout_affinities.map((pair) => ({
    ...pair,
    affinity: round6(affinityFromComponents(pair, normalizedWeights))
  }));
  const affinities = weightedPairs.map((pair) => pair.affinity);
  const p10 = percentile(affinities, 0.1);
  const p95 = percentile(affinities, 0.95);
  const displayedPairs = displayEdges(weightedPairs, Math.max(1, Math.min(20, edgeLimitPerNode)));
  const ids = network.nodes.map((node) => node.id);
  const centralityById = centrality(ids, weightedPairs);
  const clusterById = clusterIds(ids, weightedPairs);
  const edges: ProfileNetworkEdge[] = displayedPairs.map((pair) => ({
    source: pair.source,
    target: pair.target,
    affinity: pair.affinity,
    normalized_affinity: round6(normalizeValue(pair.affinity, p10, p95)),
    components: pair.components
  }));
  return {
    ...network,
    nodes: network.nodes.map((node) => ({
      ...node,
      centrality: centralityById.get(node.id) ?? node.centrality,
      cluster_id: clusterById.get(node.id) ?? node.cluster_id
    })),
    edges,
    layout_affinities: weightedPairs,
    diagnostics: {
      ...network.diagnostics,
      displayed_edge_count: edges.length,
      edge_limit_per_node: edgeLimitPerNode,
      affinity_min: affinities.length ? round6(Math.min(...affinities)) : null,
      affinity_max: affinities.length ? round6(Math.max(...affinities)) : null,
      affinity_mean: affinities.length ? round6(affinities.reduce((sum, value) => sum + value, 0) / affinities.length) : null
    },
    affinity_formula: {
      ...network.affinity_formula,
      default_weights: network.affinity_formula.default_weights,
      label: `affinity = ${normalizedWeights.personality_similarity.toFixed(2)} personality traits + ${normalizedWeights.ontology_leaf_overlap.toFixed(2)} ontology overlap + ${normalizedWeights.age_context_similarity.toFixed(2)} age/context + ${normalizedWeights.categorical_similarity.toFixed(2)} categorical demographics`
    }
  };
}
