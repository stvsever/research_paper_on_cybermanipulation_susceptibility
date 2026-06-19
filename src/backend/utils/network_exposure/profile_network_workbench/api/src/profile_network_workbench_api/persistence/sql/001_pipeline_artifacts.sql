CREATE TABLE IF NOT EXISTS schema_migrations (
    version text PRIMARY KEY,
    applied_at timestamptz NOT NULL DEFAULT now()
);

CREATE TABLE IF NOT EXISTS pipeline_runs (
    run_id text PRIMARY KEY,
    output_root text NOT NULL,
    stage_outputs_root text NOT NULL,
    config_json jsonb NOT NULL DEFAULT '{}'::jsonb,
    run_manifest_json jsonb NOT NULL DEFAULT '{}'::jsonb,
    created_at timestamptz NOT NULL DEFAULT now(),
    ingested_at timestamptz NOT NULL DEFAULT now()
);

CREATE TABLE IF NOT EXISTS pipeline_stage_artifacts (
    run_id text NOT NULL REFERENCES pipeline_runs(run_id) ON DELETE CASCADE,
    stage_id text NOT NULL,
    stage_name text NOT NULL,
    manifest_path text NOT NULL,
    primary_output_path text,
    record_count integer NOT NULL DEFAULT 0,
    created_at_utc text,
    manifest_json jsonb NOT NULL DEFAULT '{}'::jsonb,
    artifact_checksum text,
    ingested_at timestamptz NOT NULL DEFAULT now(),
    PRIMARY KEY (run_id, stage_id)
);

CREATE TABLE IF NOT EXISTS profiles (
    run_id text NOT NULL REFERENCES pipeline_runs(run_id) ON DELETE CASCADE,
    profile_id text NOT NULL,
    stage_id text NOT NULL,
    scenario_id text,
    manifest_path text NOT NULL,
    primary_output_path text,
    categorical_attributes jsonb NOT NULL DEFAULT '{}'::jsonb,
    continuous_attributes jsonb NOT NULL DEFAULT '{}'::jsonb,
    selected_leaf_nodes jsonb NOT NULL DEFAULT '[]'::jsonb,
    metadata jsonb NOT NULL DEFAULT '{}'::jsonb,
    raw_json jsonb NOT NULL DEFAULT '{}'::jsonb,
    ingested_at timestamptz NOT NULL DEFAULT now(),
    PRIMARY KEY (run_id, profile_id)
);

CREATE TABLE IF NOT EXISTS scenarios (
    run_id text NOT NULL REFERENCES pipeline_runs(run_id) ON DELETE CASCADE,
    scenario_id text NOT NULL,
    scenario_index integer NOT NULL,
    profile_id text NOT NULL,
    opinion_leaf text NOT NULL,
    attack_present boolean NOT NULL DEFAULT false,
    attack_leaf text,
    stage_id text NOT NULL,
    manifest_path text NOT NULL,
    primary_output_path text,
    metadata jsonb NOT NULL DEFAULT '{}'::jsonb,
    raw_json jsonb NOT NULL DEFAULT '{}'::jsonb,
    ingested_at timestamptz NOT NULL DEFAULT now(),
    PRIMARY KEY (run_id, scenario_id)
);

CREATE TABLE IF NOT EXISTS opinion_assessments (
    run_id text NOT NULL REFERENCES pipeline_runs(run_id) ON DELETE CASCADE,
    scenario_id text NOT NULL,
    assessment_scenario_id text,
    profile_id text NOT NULL,
    phase text NOT NULL,
    opinion_leaf text NOT NULL,
    attack_present boolean NOT NULL DEFAULT false,
    attack_leaf text,
    score integer NOT NULL,
    confidence double precision NOT NULL,
    reasoning text NOT NULL,
    model_name text NOT NULL,
    fallback_used boolean NOT NULL DEFAULT false,
    stage_id text NOT NULL,
    manifest_path text NOT NULL,
    primary_output_path text,
    assessment_json jsonb NOT NULL DEFAULT '{}'::jsonb,
    row_json jsonb NOT NULL DEFAULT '{}'::jsonb,
    ingested_at timestamptz NOT NULL DEFAULT now(),
    PRIMARY KEY (run_id, scenario_id, phase)
);

CREATE TABLE IF NOT EXISTS network_contexts (
    run_id text NOT NULL REFERENCES pipeline_runs(run_id) ON DELETE CASCADE,
    scenario_id text NOT NULL,
    profile_id text NOT NULL,
    phase text NOT NULL,
    opinion_leaf text NOT NULL,
    attack_leaf text,
    stage_id text NOT NULL,
    manifest_path text NOT NULL,
    primary_output_path text,
    context_json jsonb NOT NULL DEFAULT '{}'::jsonb,
    raw_json jsonb NOT NULL DEFAULT '{}'::jsonb,
    ingested_at timestamptz NOT NULL DEFAULT now(),
    PRIMARY KEY (run_id, scenario_id, phase)
);

CREATE TABLE IF NOT EXISTS attack_specs (
    run_id text NOT NULL REFERENCES pipeline_runs(run_id) ON DELETE CASCADE,
    scenario_id text NOT NULL,
    profile_id text,
    opinion_leaf text,
    attack_leaf text,
    stage_id text NOT NULL,
    manifest_path text NOT NULL,
    primary_output_path text,
    attack_vector_spec jsonb NOT NULL DEFAULT '{}'::jsonb,
    raw_json jsonb NOT NULL DEFAULT '{}'::jsonb,
    ingested_at timestamptz NOT NULL DEFAULT now(),
    PRIMARY KEY (run_id, scenario_id)
);

CREATE TABLE IF NOT EXISTS effectivity_deltas (
    run_id text NOT NULL REFERENCES pipeline_runs(run_id) ON DELETE CASCADE,
    scenario_id text NOT NULL,
    profile_id text,
    opinion_leaf text,
    attack_leaf text,
    baseline_score double precision,
    post_score double precision,
    delta_score double precision,
    abs_delta_score double precision,
    adversarial_effectivity double precision,
    stage_id text NOT NULL,
    manifest_path text NOT NULL,
    primary_output_path text,
    raw_json jsonb NOT NULL DEFAULT '{}'::jsonb,
    ingested_at timestamptz NOT NULL DEFAULT now(),
    PRIMARY KEY (run_id, scenario_id)
);

CREATE INDEX IF NOT EXISTS idx_scenarios_run_profile ON scenarios(run_id, profile_id);
CREATE INDEX IF NOT EXISTS idx_scenarios_run_opinion_attack ON scenarios(run_id, opinion_leaf, attack_leaf);
CREATE INDEX IF NOT EXISTS idx_assessments_run_phase ON opinion_assessments(run_id, phase);
CREATE INDEX IF NOT EXISTS idx_assessments_run_profile ON opinion_assessments(run_id, profile_id);
CREATE INDEX IF NOT EXISTS idx_contexts_run_phase ON network_contexts(run_id, phase);

