export interface LayoutNode {
  id: string;
  label: string;
  value: number;
}

export interface PositionedNode extends LayoutNode {
  x: number;
  y: number;
}

export function pairKey(source: string, target: string) {
  return source < target ? `${source}::${target}` : `${target}::${source}`;
}

function normalize(values: number[], fallback = 0.5) {
  if (!values.length) return [];
  const min = Math.min(...values);
  const max = Math.max(...values);
  const span = max - min || 1;
  return values.map((value) => (Number.isFinite(value) ? (value - min) / span : fallback));
}

function principalEigen(matrix: number[][], seed: number) {
  const n = matrix.length;
  let vector = Array.from({ length: n }, (_, index) => Math.sin(seed * (index + 1)) + 1.5);
  const norm = (items: number[]) => Math.sqrt(items.reduce((sum, value) => sum + value * value, 0)) || 1;
  vector = vector.map((value) => value / norm(vector));
  let eigenvalue = 0;
  for (let iteration = 0; iteration < 80; iteration += 1) {
    const next = matrix.map((row) => row.reduce((sum, value, index) => sum + value * vector[index], 0));
    const nextNorm = norm(next);
    vector = next.map((value) => value / nextNorm);
    eigenvalue = vector.reduce(
      (sum, value, rowIndex) =>
        sum + value * matrix[rowIndex].reduce((rowSum, item, colIndex) => rowSum + item * vector[colIndex], 0),
      0
    );
  }
  return { value: Math.max(0, eigenvalue), vector };
}

export function ringLayout(nodes: LayoutNode[], width: number, height: number): PositionedNode[] {
  const radius = Math.min(width, height) * 0.34;
  const cx = width / 2;
  const cy = height / 2;
  return nodes.map((node, index) => {
    const angle = -Math.PI / 2 + (index / Math.max(1, nodes.length)) * Math.PI * 2;
    return { ...node, x: cx + Math.cos(angle) * radius, y: cy + Math.sin(angle) * radius };
  });
}

export function mdsLayout(
  nodes: LayoutNode[],
  similarity: Map<string, number>,
  width: number,
  height: number
): PositionedNode[] {
  const n = nodes.length;
  if (n < 3) return ringLayout(nodes, width, height);
  const distancesSquared = nodes.map((source) =>
    nodes.map((target) => {
      if (source.id === target.id) return 0;
      const value = similarity.get(pairKey(source.id, target.id)) ?? 0.5;
      const bounded = Math.max(0, Math.min(1, value));
      const distance = Math.sqrt(Math.max(0, 2 * (1 - bounded)));
      return distance * distance;
    })
  );
  const rowMeans = distancesSquared.map((row) => row.reduce((sum, value) => sum + value, 0) / n);
  const columnMeans = nodes.map((_, column) => distancesSquared.reduce((sum, row) => sum + row[column], 0) / n);
  const totalMean = rowMeans.reduce((sum, value) => sum + value, 0) / n;
  const centered = distancesSquared.map((row, i) =>
    row.map((value, j) => -0.5 * (value - rowMeans[i] - columnMeans[j] + totalMean))
  );
  const first = principalEigen(centered, 1.37);
  const deflated = centered.map((row, i) => row.map((value, j) => value - first.value * first.vector[i] * first.vector[j]));
  const second = principalEigen(deflated, 2.11);
  const nx = normalize(first.vector.map((value) => value * Math.sqrt(first.value)));
  const ny = normalize(second.vector.map((value) => value * Math.sqrt(second.value)));
  const padX = Math.min(width * 0.18, 150);
  const padY = Math.min(height * 0.16, 104);
  return nodes.map((node, index) => ({
    ...node,
    x: padX + nx[index] * (width - padX * 2),
    y: padY + ny[index] * (height - padY * 2)
  }));
}
