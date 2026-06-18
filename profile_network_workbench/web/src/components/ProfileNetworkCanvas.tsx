import { useEffect, useMemo, useRef } from "react";
import { Application, Container, Graphics, Text } from "pixi.js";

import type { ProfileMeasurementResult, ProfileNetworkResponse } from "../api/types";
import { mdsLayout, pairKey, type LayoutNode } from "../lib/graph";
import type { MeasurementMode } from "./MeasurementElicitationCard";

interface AgentState {
  id: string;
  x: number;
  y: number;
  anchorX: number;
  anchorY: number;
  radius: number;
  pinned?: boolean;
}

interface ProfileNetworkCanvasProps {
  network: ProfileNetworkResponse | null;
  measurementByProfile: Map<string, ProfileMeasurementResult>;
  measurementMode: MeasurementMode;
  loading: boolean;
  error: string;
  selectedId: string;
  onSelect: (id: string) => void;
  onOpenMeasurement: (id: string) => void;
}

interface BubbleHitbox {
  profileId: string;
  x: number;
  y: number;
  width: number;
  height: number;
}

function shortLabel(label: string, limit = 18) {
  return label.length <= limit ? label : `${label.slice(0, limit - 1)}...`;
}

function measurementColor(result?: ProfileMeasurementResult) {
  if (!result) return 0x6fa8b5;
  if (Math.abs(result.score) < 75) return 0xd5ad56;
  return result.score > 0 ? 0x4ec69d : 0xf06d68;
}

function scoreLabel(score: number) {
  return score > 0 ? `+${score}` : String(score);
}

function reasoningSnippet(reasoning: string, limit = 74) {
  const compact = reasoning.replace(/\s+/g, " ").trim();
  return compact.length <= limit ? compact : `${compact.slice(0, limit - 1)}...`;
}

function measurementSubscore(result: ProfileMeasurementResult) {
  if (result.phase === "baseline" || result.delta_score === undefined) return `${Math.round(result.confidence * 100)}%`;
  if (result.phase === "post_network" && result.increment_from_private_post !== undefined) {
    const increment = result.increment_from_private_post > 0 ? `+${result.increment_from_private_post}` : String(result.increment_from_private_post);
    return `inc ${increment}`;
  }
  const delta = result.delta_score > 0 ? `+${result.delta_score}` : String(result.delta_score);
  return `d ${delta}`;
}

function edgeWeight(edge: { affinity: number; weight?: number | null; exposure_weight?: number | null }) {
  return edge.weight ?? edge.exposure_weight ?? edge.affinity;
}

function normalizedEdgeWeight(edge: { normalized_affinity?: number; normalized_weight?: number | null }) {
  return edge.normalized_weight ?? edge.normalized_affinity ?? 0;
}

function nodeExposureMetric(node: ProfileNetworkResponse["nodes"][number], key: string) {
  const assignment = node.metadata?.exposure_network_assignment;
  if (!assignment || typeof assignment !== "object" || Array.isArray(assignment)) return 0;
  const value = (assignment as Record<string, unknown>)[key];
  return typeof value === "number" ? value : 0;
}

function spreadStates(states: AgentState[], width: number, height: number) {
  const next = states.map((state) => ({ ...state }));
  const padding = 28;
  for (let iteration = 0; iteration < 90; iteration += 1) {
    for (let leftIndex = 0; leftIndex < next.length; leftIndex += 1) {
      for (let rightIndex = leftIndex + 1; rightIndex < next.length; rightIndex += 1) {
        const left = next[leftIndex];
        const right = next[rightIndex];
        let dx = right.x - left.x;
        let dy = right.y - left.y;
        let distance = Math.hypot(dx, dy);
        if (distance < 0.01) {
          dx = Math.cos((leftIndex + 1) * 2.17);
          dy = Math.sin((rightIndex + 1) * 2.17);
          distance = 1;
        }
        const minDistance = left.radius + right.radius + 18;
        if (distance >= minDistance) continue;
        const push = (minDistance - distance) * 0.52;
        const ux = dx / distance;
        const uy = dy / distance;
        left.x -= ux * push;
        left.y -= uy * push;
        right.x += ux * push;
        right.y += uy * push;
      }
    }
    next.forEach((state) => {
      state.x += (state.anchorX - state.x) * 0.012;
      state.y += (state.anchorY - state.y) * 0.012;
      state.x = Math.max(padding, Math.min(width - padding, state.x));
      state.y = Math.max(padding, Math.min(height - padding - 28, state.y));
    });
  }
  return next;
}

function buildInitialStates(network: ProfileNetworkResponse, width: number, height: number): AgentState[] {
  const nodes: LayoutNode[] = network.nodes.map((node) => ({
    id: node.id,
    label: node.label,
    value: node.centrality
  }));
  const similarity = new Map<string, number>();
  const layoutAffinities = network.layout_affinities.length ? network.layout_affinities : network.edges;
  layoutAffinities.forEach((edge) => similarity.set(pairKey(edge.source, edge.target), edgeWeight(edge)));
  const positioned = mdsLayout(nodes, similarity, width, height);
  const clusterCenters = new Map<string, { x: number; y: number; count: number }>();
  positioned.forEach((positionedNode) => {
    const source = network.nodes.find((node) => node.id === positionedNode.id);
    const clusterId = source?.cluster_id ?? "cluster_00";
    const current = clusterCenters.get(clusterId) ?? { x: 0, y: 0, count: 0 };
    current.x += positionedNode.x;
    current.y += positionedNode.y;
    current.count += 1;
    clusterCenters.set(clusterId, current);
  });
  clusterCenters.forEach((center) => {
    center.x /= Math.max(1, center.count);
    center.y /= Math.max(1, center.count);
  });
  const initial = positioned.map((node) => {
    const source = network.nodes.find((item) => item.id === node.id);
    const center = clusterCenters.get(source?.cluster_id ?? "cluster_00") ?? { x: width / 2, y: height / 2, count: 1 };
    const localScale = 1.55;
    const expandedX = center.x + (node.x - center.x) * localScale;
    const expandedY = center.y + (node.y - center.y) * localScale;
    const anchorX = Math.max(36, Math.min(width - 36, expandedX));
    const anchorY = Math.max(36, Math.min(height - 64, expandedY));
    return {
    id: node.id,
    x: anchorX,
    y: anchorY,
    anchorX,
    anchorY,
    radius: 7 + Math.sqrt(Math.max(0, node.value)) * 8
  };
  });
  return spreadStates(initial, width, height);
}

function drawGrid(graphics: Graphics, width: number, height: number) {
  graphics.rect(0, 0, width, height).fill({ color: 0x09131a, alpha: 1 });
  graphics.rect(0, 0, width, height).fill({ color: 0x102331, alpha: 0.72 });
  graphics.stroke({ color: 0x476275, alpha: 0.2, width: 1 });
  const step = 56;
  for (let x = step; x < width; x += step) graphics.moveTo(x, 0).lineTo(x, height);
  for (let y = step; y < height; y += step) graphics.moveTo(0, y).lineTo(width, y);
  graphics.stroke({ color: 0x95c7d1, alpha: 0.12, width: 1 });
  graphics.moveTo(width / 2, 0).lineTo(width / 2, height);
  graphics.moveTo(0, height / 2).lineTo(width, height / 2);
}

function drawArrowHead(graphics: Graphics, source: AgentState, target: AgentState, color: number, alpha: number, width: number) {
  const dx = target.x - source.x;
  const dy = target.y - source.y;
  const distance = Math.hypot(dx, dy);
  if (distance < 1) return;
  const angle = Math.atan2(dy, dx);
  const tipX = target.x - Math.cos(angle) * (target.radius + 3);
  const tipY = target.y - Math.sin(angle) * (target.radius + 3);
  const arrowLength = Math.max(5, Math.min(10, width * 3.2));
  const spread = 0.55;
  graphics.moveTo(tipX, tipY);
  graphics.lineTo(tipX - Math.cos(angle - spread) * arrowLength, tipY - Math.sin(angle - spread) * arrowLength);
  graphics.moveTo(tipX, tipY);
  graphics.lineTo(tipX - Math.cos(angle + spread) * arrowLength, tipY - Math.sin(angle + spread) * arrowLength);
  graphics.stroke({ color, alpha, width: Math.max(0.7, width * 0.55) });
}

function drawNetwork(
  graphics: Graphics,
  labelLayer: Container,
  states: AgentState[],
  network: ProfileNetworkResponse,
  measurementByProfile: Map<string, ProfileMeasurementResult>,
  selectedId: string,
  measurementMode: MeasurementMode,
  width: number,
  height: number
): BubbleHitbox[] {
  graphics.clear();
  drawGrid(graphics, width, height);
  const stateById = new Map(states.map((state) => [state.id, state]));
  const nodeById = new Map(network.nodes.map((node) => [node.id, node]));
  const clusterById = new Map(network.nodes.map((node) => [node.id, node.cluster_id]));
  const affinityValues = network.layout_affinities.map((edge) => edgeWeight(edge));
  const minAffinity = affinityValues.length ? Math.min(...affinityValues) : 0;
  const maxAffinity = affinityValues.length ? Math.max(...affinityValues) : 1;
  const normalizeAffinity = (value: number) => {
    const span = maxAffinity - minAffinity || 1;
    return Math.max(0, Math.min(1, (value - minAffinity) / span));
  };
  const selectedNeighbors = new Set<string>();
  network.edges.forEach((edge) => {
    if (edge.source === selectedId) selectedNeighbors.add(edge.target);
    if (edge.target === selectedId) selectedNeighbors.add(edge.source);
  });

  network.layout_affinities.forEach((edge) => {
    const source = stateById.get(edge.source);
    const target = stateById.get(edge.target);
    if (!source || !target) return;
    const sameCluster = clusterById.get(edge.source) === clusterById.get(edge.target);
    const normalized = normalizeAffinity(edgeWeight(edge));
    graphics.moveTo(source.x, source.y);
    graphics.lineTo(target.x, target.y);
    graphics.stroke({
      color: sameCluster ? 0x7fb7c5 : 0x668696,
      alpha: sameCluster ? 0.012 + normalized * 0.035 : 0.005 + normalized * 0.012,
      width: sameCluster ? 0.25 + normalized * 0.55 : 0.16 + normalized * 0.22
    });
  });

  network.edges.forEach((edge) => {
    const source = stateById.get(edge.source);
    const target = stateById.get(edge.target);
    if (!source || !target) return;
    const active = !selectedId || edge.source === selectedId || edge.target === selectedId;
    const normalized = normalizedEdgeWeight(edge);
    graphics.moveTo(source.x, source.y);
    graphics.lineTo(target.x, target.y);
    graphics.stroke({
      color: 0x81b5c5,
      alpha: active ? 0.08 + normalized * 0.35 : 0.035,
      width: active ? 0.6 + normalized * 2.2 : 0.45
    });
    if (edge.directed) {
      drawArrowHead(graphics, source, target, 0x9bc9d4, active ? 0.28 + normalized * 0.38 : 0.08, active ? 0.6 + normalized * 2.2 : 0.45);
    }
  });

  labelLayer.removeChildren();
  const topIds = new Set(
    network.nodes
      .slice()
      .sort((left, right) => right.centrality - left.centrality)
      .slice(0, 10)
      .map((node) => node.id)
  );
  if (selectedId) topIds.add(selectedId);

  const latestBubbleIds = new Set(
    Array.from(measurementByProfile.values())
      .slice(-7)
      .map((result) => result.profile_id)
  );
  if (selectedId && measurementByProfile.has(selectedId)) latestBubbleIds.add(selectedId);
  const bubbleHitboxes: BubbleHitbox[] = [];

  const incomingValues = network.nodes.map((node) => nodeExposureMetric(node, "incoming_exposure_weight"));
  const maxIncoming = Math.max(1, ...incomingValues);

  states.forEach((state) => {
    const node = nodeById.get(state.id);
    if (!node) return;
    const measurement = measurementByProfile.get(state.id);
    const selected = selectedId === state.id;
    const neighbor = selectedNeighbors.has(state.id);
    const muted = selectedId && !selected && !neighbor;
    const color = measurementColor(measurement);
    const alpha = muted ? 0.34 : 0.92;
    const halo = selected ? 8 : measurement ? 4 : 2;
    const incomingNorm = Math.max(0, Math.min(1, nodeExposureMetric(node, "incoming_exposure_weight") / maxIncoming));
    const strokeWidth = selected ? 2.4 : 0.9 + incomingNorm * 2.1;

    graphics.circle(state.x, state.y, state.radius + halo).fill({ color, alpha: selected ? 0.22 : 0.08 });
    graphics.circle(state.x, state.y, state.radius).fill({ color, alpha });
    graphics.circle(state.x, state.y, state.radius).stroke({
      color: selected ? 0xf0c15c : 0xb9d9dd,
      alpha: selected ? 0.95 : 0.36 + incomingNorm * 0.34,
      width: strokeWidth
    });

    if (measurement && latestBubbleIds.has(state.id)) {
      const bubbleWidth = selected ? 232 : 192;
      const bubbleHeight = selected ? 64 : 50;
      const placeLeft = state.x + state.radius + bubbleWidth + 18 > width;
      const bubbleX = placeLeft ? state.x - state.radius - bubbleWidth - 12 : state.x + state.radius + 12;
      const bubbleY = Math.max(12, Math.min(height - bubbleHeight - 34, state.y - bubbleHeight - 12));
      const bubbleColor = measurementColor(measurement);
      bubbleHitboxes.push({ profileId: state.id, x: bubbleX, y: bubbleY, width: bubbleWidth, height: bubbleHeight });

      graphics.moveTo(state.x, state.y);
      graphics.lineTo(placeLeft ? bubbleX + bubbleWidth : bubbleX, bubbleY + bubbleHeight * 0.56);
      graphics.stroke({ color: bubbleColor, alpha: selected ? 0.62 : 0.34, width: selected ? 1.8 : 1.1 });
      graphics.rect(bubbleX, bubbleY, bubbleWidth, bubbleHeight).fill({ color: 0x0d1920, alpha: selected ? 0.94 : 0.84 });
      graphics.rect(bubbleX, bubbleY, bubbleWidth, bubbleHeight).stroke({ color: bubbleColor, alpha: selected ? 0.92 : 0.52, width: selected ? 1.6 : 1 });

      graphics.rect(bubbleX, bubbleY, selected ? 56 : 48, bubbleHeight).fill({ color: bubbleColor, alpha: selected ? 0.22 : 0.16 });
      graphics.moveTo(bubbleX + (selected ? 56 : 48), bubbleY + 6);
      graphics.lineTo(bubbleX + (selected ? 56 : 48), bubbleY + bubbleHeight - 6);
      graphics.stroke({ color: bubbleColor, alpha: 0.38, width: 1 });

      const score = new Text({
        text: scoreLabel(measurement.score),
        style: {
          fill: 0xffffff,
          fontFamily: "Inter, system-ui, sans-serif",
          fontSize: selected ? 15 : 13,
          fontWeight: "900"
        }
      });
      score.x = bubbleX + 7;
      score.y = bubbleY + (selected ? 12 : 10);
      labelLayer.addChild(score);

      const confidence = new Text({
        text: measurementSubscore(measurement),
        style: {
          fill: 0xc9dce1,
          fontFamily: "Inter, system-ui, sans-serif",
          fontSize: 9,
          fontWeight: "800"
        }
      });
      confidence.x = bubbleX + 9;
      confidence.y = bubbleY + (selected ? 36 : 31);
      labelLayer.addChild(confidence);

      const title = new Text({
        text: node.label,
        style: {
          fill: 0xf4fbfd,
          fontFamily: "Inter, system-ui, sans-serif",
          fontSize: selected ? 12 : 10,
          fontWeight: "800"
        }
      });
      title.x = bubbleX + (selected ? 66 : 58);
      title.y = bubbleY + 6;
      labelLayer.addChild(title);

      const body = new Text({
        text: reasoningSnippet(
          measurementMode === "network" && measurement.delta_score !== undefined
            ? `Network delta ${measurement.delta_score > 0 ? "+" : ""}${measurement.delta_score}. ${measurement.reasoning}`
            : measurementMode === "post" && measurement.delta_score !== undefined
            ? `Post delta ${measurement.delta_score > 0 ? "+" : ""}${measurement.delta_score}. ${measurement.reasoning}`
            : measurementMode === "post_network" && measurement.increment_from_private_post !== undefined
            ? `Post-network increment ${measurement.increment_from_private_post > 0 ? "+" : ""}${measurement.increment_from_private_post}. ${measurement.reasoning}`
            : measurement.reasoning,
          selected ? 92 : 62
        ),
        style: {
          fill: 0xbfd0d5,
          fontFamily: "Inter, system-ui, sans-serif",
          fontSize: selected ? 10 : 9,
          fontWeight: "500",
          wordWrap: true,
          wordWrapWidth: bubbleWidth - 18
        }
      });
      body.x = bubbleX + (selected ? 66 : 58);
      body.y = bubbleY + (selected ? 25 : 23);
      labelLayer.addChild(body);
    }

    if (topIds.has(state.id)) {
      const label = new Text({
        text: shortLabel(node.label),
        style: {
          fill: selected ? 0xffffff : 0xd9e8ed,
          fontFamily: "Inter, system-ui, sans-serif",
          fontSize: selected ? 13 : 11,
          fontWeight: selected ? "700" : "600"
        }
      });
      label.x = state.x + state.radius + 6;
      label.y = state.y - 8;
      labelLayer.addChild(label);
    }
  });
  return bubbleHitboxes;
}

export function ProfileNetworkCanvas({
  network,
  measurementByProfile,
  measurementMode,
  loading,
  error,
  selectedId,
  onSelect,
  onOpenMeasurement
}: ProfileNetworkCanvasProps) {
  const containerRef = useRef<HTMLDivElement | null>(null);
  const appRef = useRef<Application | null>(null);
  const statesRef = useRef<AgentState[]>([]);
  const networkRef = useRef<ProfileNetworkResponse | null>(network);
  const selectedRef = useRef(selectedId);
  const measurementRef = useRef(measurementByProfile);
  const measurementModeRef = useRef(measurementMode);
  const draggingRef = useRef("");
  const bubbleHitboxesRef = useRef<BubbleHitbox[]>([]);

  const edgeSignature = useMemo(
    () => network?.edges.map((edge) => `${edge.source}-${edge.target}-${edgeWeight(edge)}`).join("|") ?? "",
    [network]
  );

  useEffect(() => {
    networkRef.current = network;
  }, [network]);

  useEffect(() => {
    selectedRef.current = selectedId;
  }, [selectedId]);

  useEffect(() => {
    measurementRef.current = measurementByProfile;
  }, [measurementByProfile]);

  useEffect(() => {
    measurementModeRef.current = measurementMode;
  }, [measurementMode]);

  useEffect(() => {
    if (!containerRef.current || !network) return;
    const rect = containerRef.current.getBoundingClientRect();
    statesRef.current = buildInitialStates(network, Math.max(720, rect.width), Math.max(520, rect.height));
  }, [network, edgeSignature]);

  useEffect(() => {
    if (!containerRef.current) return;
    const container = containerRef.current;
    const app = new Application();
    const graphics = new Graphics();
    const labelLayer = new Container();
    let disposed = false;

    async function mount() {
      await app.init({
        antialias: true,
        autoDensity: true,
        backgroundAlpha: 0,
        resolution: Math.min(window.devicePixelRatio || 1, 2),
        resizeTo: container
      });
      if (disposed) {
        app.destroy();
        return;
      }
      appRef.current = app;
      app.stage.addChild(graphics);
      app.stage.addChild(labelLayer);
      container.appendChild(app.canvas);

      const onPointerDown = (event: PointerEvent) => {
        const bounds = app.canvas.getBoundingClientRect();
        const x = event.clientX - bounds.left;
        const y = event.clientY - bounds.top;
        const bubble = bubbleHitboxesRef.current.find(
          (item) => x >= item.x && x <= item.x + item.width && y >= item.y && y <= item.y + item.height
        );
        if (bubble) {
          draggingRef.current = "";
          onSelect(bubble.profileId);
          onOpenMeasurement(bubble.profileId);
          return;
        }
        let closest = "";
        let distance = Number.POSITIVE_INFINITY;
        statesRef.current.forEach((state) => {
          const nextDistance = Math.hypot(state.x - x, state.y - y);
          if (nextDistance <= state.radius + 13 && nextDistance < distance) {
            closest = state.id;
            distance = nextDistance;
          }
        });
        draggingRef.current = closest;
        onSelect(closest);
      };
      const onPointerMove = (event: PointerEvent) => {
        if (!draggingRef.current) return;
        const bounds = app.canvas.getBoundingClientRect();
        const state = statesRef.current.find((item) => item.id === draggingRef.current);
        if (!state) return;
        state.x = event.clientX - bounds.left;
        state.y = event.clientY - bounds.top;
      };
      const onPointerUp = () => {
        draggingRef.current = "";
      };

      app.canvas.addEventListener("pointerdown", onPointerDown);
      app.canvas.addEventListener("pointermove", onPointerMove);
      window.addEventListener("pointerup", onPointerUp);

      app.ticker.add(() => {
        const activeNetwork = networkRef.current;
        const screen = app.renderer.screen;
        if (!activeNetwork) {
          graphics.clear();
          drawGrid(graphics, screen.width, screen.height);
          return;
        }
        if (!statesRef.current.length) {
          statesRef.current = buildInitialStates(activeNetwork, screen.width, screen.height);
        }
        bubbleHitboxesRef.current = drawNetwork(
          graphics,
          labelLayer,
          statesRef.current,
          activeNetwork,
          measurementRef.current,
          selectedRef.current,
          measurementModeRef.current,
          screen.width,
          screen.height
        );
      });

      return () => {
        app.canvas.removeEventListener("pointerdown", onPointerDown);
        app.canvas.removeEventListener("pointermove", onPointerMove);
        window.removeEventListener("pointerup", onPointerUp);
      };
    }

    let cleanup: (() => void) | undefined;
    mount().then((nextCleanup) => {
      cleanup = nextCleanup;
    });

    return () => {
      disposed = true;
      cleanup?.();
      appRef.current = null;
      if (app.canvas.parentNode) app.canvas.parentNode.removeChild(app.canvas);
      app.destroy();
    };
  }, [network, onSelect, onOpenMeasurement]);

  if (loading) {
    return (
      <div className="canvas-state">
        <strong>Loading profile network</strong>
        <span>Loading profile positions and exposure edges.</span>
      </div>
    );
  }
  if (error) {
    return (
      <div className="canvas-state warning">
        <strong>Network unavailable</strong>
        <span>{error}</span>
      </div>
    );
  }
  if (!network) {
    return (
      <div className="canvas-state">
        <strong>No network loaded</strong>
        <span>Refresh the workbench to reconstruct the profile panel.</span>
      </div>
    );
  }
  return <div ref={containerRef} className="profile-network-canvas" />;
}
