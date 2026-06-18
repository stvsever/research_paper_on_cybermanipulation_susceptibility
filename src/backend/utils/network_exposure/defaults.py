from __future__ import annotations

DEFAULT_GRAPH_ID = "politisky24_bluesky_v1"
DEFAULT_ASSIGNMENT_MODE = "ranked_seeded_profiles"
DEFAULT_MAX_EXEMPLARS = 8

EDGE_FILE_FULL = "edges_full.csv"
EDGE_FILE_PROMPT_TOP30 = "edges_prompt_top30.csv"

EDGE_DIRECTION = "source_position_id -> target_position_id"
EDGE_MEANING = "visible_peer_to_exposed_receiver"

REQUIRED_PACKAGE_FILES = (
    "manifest.json",
    "data_dictionary.json",
    "nodes.csv",
    "node_metrics.csv",
    "neighborhood_metrics.csv",
    "propagation_metrics.csv",
    "assignment_positions.csv",
    EDGE_FILE_FULL,
    EDGE_FILE_PROMPT_TOP30,
)

REQUIRED_EDGE_COLUMNS = (
    "source_position_id",
    "target_position_id",
    "raw_weight",
    "total_events",
    "interaction_types",
    "exposure_weight",
    "rank_for_receiver",
)

REQUIRED_POSITION_COLUMNS = ("position_id",)
REQUIRED_ASSIGNMENT_COLUMNS = ("assignment_rank", "position_id")

