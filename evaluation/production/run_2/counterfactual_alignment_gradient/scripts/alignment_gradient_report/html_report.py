from __future__ import annotations

"""HTML report rendering for the alignment-gradient branch analysis.

This module keeps the narrative report template separate from data preparation, modeling, and figure generation.
"""

import html
from pathlib import Path

import pandas as pd

from .formatting import _display_path, _fmt, _fmt_p, _relative_src


def _table_html(df: pd.DataFrame, rows: int = 12, columns: list[str] | None = None) -> str:
    view = df.copy()
    if columns:
        view = view[[col for col in columns if col in view.columns]]
    return view.head(rows).to_html(index=False, classes="data-table", border=0, escape=True)


def _figure_html(
    *,
    number: int,
    title: str,
    explanation: str,
    path: Path | None,
    reports_dir: Path,
    alt: str,
    note: str,
    heading_prefix: str = "Main Figure",
) -> str:
    if path is None:
        return ""
    heading = f"{heading_prefix} {number}. {html.escape(title)}"
    return f"""
  <h3>{heading}</h3>
  <p>{explanation}</p>
  <figure>
    <img src="{_relative_src(path, reports_dir)}" alt="{html.escape(alt)}">
    <figcaption><strong>Note.</strong> {note}</figcaption>
  </figure>
"""


def _render_html(
    *,
    run_id: str,
    branch_root: Path,
    report_mode: str,
    condition: pd.DataFrame,
    level: pd.DataFrame,
    models: pd.DataFrame,
    robustness: pd.DataFrame,
    design_balance: pd.DataFrame,
    original_comparison: pd.DataFrame,
    quality: pd.DataFrame,
    figures: dict[str, Path],
    reports_dir: Path,
) -> Path:
    report_path = reports_dir / f"{run_id}_alignment_gradient_h3h4.html"
    pn_model = models[
        models["outcome"].eq("mean_pn_increment_effectivity") & models["model_id"].eq("main_fe_ols")
    ].iloc[0]
    total_model = models[
        models["outcome"].eq("mean_ae_total_network") & models["model_id"].eq("main_fe_ols")
    ].iloc[0]
    max_alignment_error = float(condition["alignment_abs_error"].max())
    mean_alignment_error = float(condition["alignment_abs_error"].mean())
    stage04b_fallbacks = quality.loc[quality["metric"].eq("stage04b_fallback_count"), "value"].iloc[0]
    stage04b_skips = quality.loc[quality["metric"].eq("stage04b_skipped_count"), "value"].iloc[0]
    target_corr = float(condition["target_alignment_z"].corr(condition["achieved_alignment_z"]))
    design_summary = pd.DataFrame(
        [
            {"Design element": "Profiles per condition", "Value": int(condition["n_profiles"].min())},
            {"Design element": "Opinion x attack condition cells", "Value": int(condition.shape[0])},
            {"Design element": "Alignment targets", "Value": ", ".join(f"{v:+.2f}" for v in sorted(condition["target_alignment_z"].unique()))},
            {"Design element": "Cells per alignment target", "Value": int(level["n_conditions"].min())},
            {"Design element": "Primary predictor", "Value": "achieved_alignment_z"},
            {"Design element": "Primary outcome", "Value": "mean((PN - P) x d)"},
        ]
    )
    measurement = pd.DataFrame(
        [
            {"Symbol": "B", "Measurement state": "Private baseline opinion", "Pipeline field": "baseline_score", "Analytical role": "Pre-attack private reference point"},
            {"Symbol": "BN", "Measurement state": "Baseline after empirical incoming peer context", "Pipeline field": "network_exposure_score", "Analytical role": "Pre-attack network-context check"},
            {"Symbol": "P", "Measurement state": "Private post-attack opinion", "Pipeline field": "post_score", "Analytical role": "Private attack response"},
            {"Symbol": "PN", "Measurement state": "Post-attack after same-condition incoming peer context", "Pipeline field": "post_attack_network_score", "Analytical role": "Final network-exposed state"},
        ]
    )
    deltas = pd.DataFrame(
        [
            {"Quantity": "Private susceptibility", "Formula": "AE_private = (P - B) x d", "Question answered": "How much did one profile privately move toward the attack goal?"},
            {"Quantity": "Post-network effectivity", "Formula": "PN_increment_effectivity = (PN - P) x d", "Question answered": "Does the post-attack peer context amplify or dampen the attack direction?"},
            {"Quantity": "Total network attack effect", "Formula": "AE_total_network = (PN - B) x d", "Question answered": "What is the final attack-aligned effect after private and network exposure?"},
            {"Quantity": "Sender-reach susceptibility alignment", "Formula": "weighted mean(AE_private) - unweighted mean(AE_private), z-scaled", "Question answered": "Are high-reach senders more susceptible or more resilient than the condition average?"},
        ]
    )
    hypotheses = pd.DataFrame(
        [
            {
                "Hypothesis": "H3. Central susceptible sender amplification",
                "Operational test": "Higher achieved_alignment_z predicts higher mean_pn_increment_effectivity.",
                "Expected direction": "Positive coefficient",
            },
            {
                "Hypothesis": "H4. Central resilient sender attenuation",
                "Operational test": "Lower achieved_alignment_z predicts lower mean_pn_increment_effectivity and lower final network effect.",
                "Expected direction": "Same positive coefficient, interpreted from the negative-alignment side",
            },
        ]
    )
    main_figures = "\n".join(
        [
            _figure_html(
                number=1,
                title="Alignment Manipulation Check",
                explanation=(
                    "This figure verifies that the branch successfully turns the original fixed profile-position assignment "
                    "into a controlled alignment-gradient manipulation. The nominal target levels are design anchors; the "
                    "continuous achieved alignment is the analysis variable."
                ),
                path=figures.get("target_vs_achieved"),
                reports_dir=reports_dir,
                alt="Target versus achieved alignment",
                note=(
                    "Each point is one condition cell. The x-axis is the planned sender-reach susceptibility alignment; "
                    "the y-axis is the achieved alignment after deterministic assignment optimization. Each target appears "
                    "in five cells, and each attack receives all seven target levels once. Inferential models use "
                    "<code>achieved_alignment_z</code>, not the nominal level label."
                ),
            ),
            _figure_html(
                number=2,
                title="Condition-Specific Vulnerability Planes",
                explanation=(
                    "This figure is the direct diagnostic for the counterfactual mechanism. It shows, per condition, "
                    "whether high-reach sender positions are occupied by profiles that are privately more susceptible or "
                    "more resilient than the condition average."
                ),
                path=figures.get("condition_vulnerability_planes"),
                reports_dir=reports_dir,
                alt="Condition-specific vulnerability planes",
                note=(
                    "Each panel is one <code>opinion x attack</code> condition. The x-axis is within-condition "
                    "<code>AE_private</code> z-score; the y-axis is direct sender-reach percentile. The shaded band marks "
                    "the top 20 percent of sender reach, and outlined points mark the top 10 percent. Color uses a fixed "
                    "blue-orange signed scale centered at zero: blue means below-condition-average susceptibility or "
                    "resilience, and orange means above-condition-average susceptibility. No fitted line is shown because "
                    "the figure visualizes assignment placement, not a within-panel linear model. The panel label "
                    "<code>sender alignment z</code> is the achieved sender-reach susceptibility alignment: positive "
                    "values mean high-reach positions are shifted toward susceptible profiles, and negative values mean "
                    "high-reach positions are shifted toward resilient profiles."
                ),
            ),
            _figure_html(
                number=3,
                title="Full 35-Condition Network Overlay",
                explanation=(
                    "This figure adapts the original Run 2 network overlay to the branch. The empirical exposure graph is "
                    "held fixed, while profile-position assignments vary by condition to create the alignment gradient. "
                    "The edge layer is shown as restrained graph context: faint lines preserve the induced exposure "
                    "substrate, and sparse arrows make direction visible without overwhelming the susceptibility pattern."
                ),
                path=figures.get("full_network_overlay"),
                reports_dir=reports_dir,
                alt="Full 35-condition branch network overlay",
                note=(
                    "Each panel is one branch condition. Node coordinates and directed empirical edges are the same in "
                    "every panel. Faint lines show all induced <code>edges_prompt_top30</code> exposure edges among the "
                    "100 empirical positions. Darker arrows show the sparse directed backbone, selected as the strongest "
                    "incoming edge per receiver plus the global top 50 edges by <code>exposure_weight</code>; arrows point "
                    "from <code>visible peer -&gt; exposed receiver</code>. Node size is direct sender-reach percentile "
                    "based on <code>outgoing_visibility_weight</code>. Node color is the within-condition percentile of "
                    "<code>AE_private = (P - B) x d</code>: blue indicates lower relative susceptibility or resilience, "
                    "neutral is the condition median, and orange indicates higher relative susceptibility. Colors compare "
                    "profiles within a condition, not absolute effect magnitude across conditions. The panel label "
                    "<code>sender alignment z</code> is the achieved sender-reach susceptibility alignment: positive "
                    "values indicate susceptible high-reach placement, and negative values indicate resilient high-reach "
                    "placement."
                ),
            ),
            _figure_html(
                number=4,
                title="Alignment-Gradient Outcome Test",
                explanation=(
                    "This figure is the primary H3/H4 mechanism test. In the counterfactual alignment-gradient branch, "
                    "higher sender-reach susceptibility alignment significantly increases post-network amplification, "
                    "supporting H3. The same positive slope also means that placing resilient profiles in high-reach "
                    "sender positions attenuates network-wide attack effects, supporting H4. The visual removes opinion "
                    "and attack fixed effects from both alignment and outcome so the plotted relationship matches the "
                    "condition-level fixed-effect model used for the statistical claim."
                ),
                path=figures.get("alignment_outcomes"),
                reports_dir=reports_dir,
                alt="Fixed-effect adjusted alignment-gradient outcome test",
                note=(
                    "Each point is one <code>opinion x attack</code> condition cell, averaging 100 profiles. "
                    "<code>B</code> is private baseline, <code>P</code> is private post-attack, <code>PN</code> is "
                    "post-attack after same-condition incoming peer exposure, and <code>d</code> is the adversarial "
                    "direction. The left panel is the primary network-mechanism outcome "
                    "<code>mean((PN - P) x d)</code>; the right panel is the secondary final endpoint "
                    "<code>mean((PN - B) x d)</code>. Axes are fixed-effect residuals: units on the y-axis remain "
                    "attack-aligned opinion-score points after removing opinion and attack fixed effects, and units on "
                    "the x-axis are adjusted sender-reach susceptibility alignment z-values after the same fixed "
                    "effects; negative values indicate relatively resilient high-reach placement and positive values "
                    "indicate relatively susceptible high-reach placement. The line is the HC3 fixed-effect coefficient, "
                    "and the shaded wedge is the 95% HC3 "
                    "coefficient interval, not a prediction interval. The directional permutation p-value permutes "
                    "alignment within attack families. Benjamini-Hochberg FDR q-values across the two displayed "
                    "endpoints are reported in the Figure 4 summary table as a sensitivity check; the planned primary "
                    "claim is the left-panel HC3 test."
                ),
            ),
            _figure_html(
                number=5,
                title="Marginal Means And Robustness",
                explanation=(
                    "The marginal means provide an ordered descriptive check across the seven design anchors. The "
                    "robustness panel tests whether the achieved-alignment coefficient survives robust standard errors "
                    "and targeted sensitivity exclusions."
                ),
                path=figures.get("level_means"),
                reports_dir=reports_dir,
                alt="Marginal means by alignment level",
                note=(
                    "Points are alignment-target means across five condition cells; vertical intervals are standard "
                    "errors across condition cells. These means are descriptive because the quantitative claim is made "
                    "with continuous <code>achieved_alignment_z</code>. The companion robustness table below reports "
                    "classical FE, HC3, within-attack permutation, fallback-row exclusion, and heuristic-warning "
                    "condition exclusion checks."
                ),
            ),
        ]
    )
    robustness_figure = _figure_html(
        number=1,
        title="Robustness Coefficients",
        explanation=(
            "This diagnostic is not a separate hypothesis figure. It summarizes whether the achieved-alignment coefficient "
            "is stable across the planned robustness checks."
        ),
        path=figures.get("robustness_coefficients"),
        reports_dir=reports_dir,
        alt="Robustness coefficients for H3/H4",
        note=(
            "Intervals are coefficient plus or minus 1.96 standard errors where a standard error is defined. The "
            "permutation p-values are reported in the robustness table because they do not produce a standard-error "
            "interval. All models remain condition-level; profile rows are not treated as independent H3/H4 units."
        ),
        heading_prefix="Diagnostic Figure",
    )
    quality_figure = _figure_html(
        number=2,
        title="Quality Gate Summary",
        explanation=(
            "This final diagnostic verifies that the branch rerun and merged analysis are complete enough to support the "
            "mechanism report."
        ),
        path=figures.get("quality_gates"),
        reports_dir=reports_dir,
        alt="Quality gates",
        note=(
            "The branch requires 3,500 merged Stage 04b rows, 3,500 Stage 05 rows, 35 condition cells, zero Stage 04b "
            "fallbacks, zero Stage 04b skipped tasks, and maximum absolute target-achieved alignment error below 0.03. "
            "All gates must pass before interpreting the H3/H4 mechanism figures."
        ),
        heading_prefix="Diagnostic Figure",
    )
    html_text = f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <title>{html.escape(run_id)} Alignment-Gradient H3/H4 Report</title>
  <style>
    body {{ font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; margin: 34px; color: #172033; background: #fbfcfe; line-height: 1.48; }}
    h1, h2, h3 {{ color: #111827; }}
    h1 {{ margin-bottom: 6px; }}
    h2 {{ margin-top: 36px; border-top: 1px solid #d9deea; padding-top: 22px; }}
    h3 {{ margin-top: 28px; }}
    p {{ max-width: 1120px; }}
    .note, figcaption {{ color: #566174; max-width: 1120px; line-height: 1.5; }}
    .grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(230px, 1fr)); gap: 12px; margin: 18px 0; max-width: 1120px; }}
    .metric {{ border: 1px solid #d9deea; background: white; padding: 13px; border-radius: 8px; }}
    .metric-label {{ font-size: 11px; color: #6b7280; text-transform: uppercase; letter-spacing: .03em; }}
    .metric-value {{ font-size: 24px; font-weight: 650; margin-top: 4px; }}
    figure {{ margin: 14px 0 30px; padding: 16px; border: 1px solid #d9deea; border-radius: 8px; background: white; }}
    figure img {{ max-width: 100%; display: block; margin: 0 auto 12px; }}
    .data-table {{ border-collapse: collapse; font-size: 13px; width: 100%; background: white; margin: 12px 0 22px; }}
    .data-table th, .data-table td {{ border-bottom: 1px solid #e5e7eb; padding: 7px 8px; text-align: left; vertical-align: top; }}
    .data-table th {{ background: #f3f6fb; color: #374151; }}
    code {{ background: #eef2f7; padding: 1px 4px; border-radius: 4px; }}
    .callout {{ border-left: 4px solid #2E4780; background: #f3f6fb; padding: 12px 14px; max-width: 1120px; }}
  </style>
</head>
<body>
  <h1>Counterfactual Alignment-Gradient H3/H4 Network Exposure Report</h1>
  <p>
    This branch uses the already-paid private post-attack measurements from production Run 2 and experimentally
    reassigns profiles to empirical exposure-network positions by condition. The aim is narrow: create a controlled
    sender-reach susceptibility alignment gradient so H3 and H4 can be tested as network-position mechanisms rather than
    as incidental correlations from the original fixed assignment.
  </p>
  <p>
    The original production Run 2 remains the source for H1/H2, private susceptibility, and observed-assignment analyses.
    This counterfactual branch is the primary experimental test for whether susceptible or resilient profiles occupying
    high-reach sender positions amplifies or attenuates post-attack network effects.
  </p>
  <p class="note">Report mode: <code>{html.escape(report_mode)}</code>; branch root: <code>{html.escape(_display_path(branch_root))}</code>.</p>
  <div class="grid">
    <div class="metric"><div class="metric-label">Condition cells</div><div class="metric-value">{condition.shape[0]}</div></div>
    <div class="metric"><div class="metric-label">Profiles per cell</div><div class="metric-value">{int(condition["n_profiles"].min())}</div></div>
    <div class="metric"><div class="metric-label">PN FE slope</div><div class="metric-value">{_fmt(pn_model["estimate"])}</div></div>
    <div class="metric"><div class="metric-label">PN p-value</div><div class="metric-value">{_fmt_p(pn_model["p_value"])}</div></div>
    <div class="metric"><div class="metric-label">Total FE slope</div><div class="metric-value">{_fmt(total_model["estimate"])}</div></div>
    <div class="metric"><div class="metric-label">Stage 04b fallbacks/skips</div><div class="metric-value">{int(stage04b_fallbacks)}/{int(stage04b_skips)}</div></div>
  </div>
  <div class="callout">
    <strong>Primary result.</strong> The planned condition-level model estimates the effect of continuous
    <code>achieved_alignment_z</code> while controlling for opinion and attack fixed effects. For the primary mechanism
    endpoint <code>mean((PN - P) x d)</code>, the estimated alignment coefficient is
    <strong>{_fmt(pn_model["estimate"])}</strong> score points per one alignment-z unit
    (<code>p = {_fmt_p(pn_model["p_value"])}</code>). For the broader final endpoint
    <code>mean((PN - B) x d)</code>, the coefficient is <strong>{_fmt(total_model["estimate"])}</strong>
    (<code>p = {_fmt_p(total_model["p_value"])}</code>).
  </div>

  <h2>1. Experimental Design</h2>
  <p>
    The branch has seven quasi-continuous alignment targets and five replicated cells per target, yielding 35 condition
    cells and 3,500 profile-level Stage 04b assessments. The empirical exposure graph and sender-reach distribution stay
    fixed. What varies by condition is which generated profile occupies which empirical network position.
  </p>
  {_table_html(design_summary, rows=20)}
  <p>
    The achieved manipulation is tight: target-achieved correlation is <code>{target_corr:.4f}</code>, mean absolute
    target error is <code>{mean_alignment_error:.4f}</code>, and maximum absolute target error is
    <code>{max_alignment_error:.4f}</code>.
  </p>
  {_table_html(design_balance, rows=10, columns=["alignment_level", "target_alignment_z", "n_conditions", "n_attacks", "n_opinions", "mean_achieved_alignment_z", "max_alignment_abs_error"])}

  <h2>2. Measurement Backbone</h2>
  <p>
    The branch keeps the original four-state measurement structure. The manipulated assignment affects only the
    post-attack network-exposure context used for <code>PN</code>; it does not rerun private baseline or private
    post-attack opinions.
  </p>
  {_table_html(measurement, rows=10)}
  <h3>Core deltas</h3>
  {_table_html(deltas, rows=10)}

  <h2>3. Hypothesis Mapping</h2>
  <p>
    The statistical unit for H3/H4 is the condition cell, not the profile row. Profile rows are used to compute
    condition-level susceptibility placement and condition-level network outcomes.
  </p>
  {_table_html(hypotheses, rows=10)}

  <h2>4. Main Figures: Alignment-Gradient Network Exposure Mechanism</h2>
  <p>
    The figures follow the same logic as the original Run 2 network-exposure report: first verify the manipulation,
    then show where susceptibility lands on sender-reach positions, then test whether achieved alignment predicts the
    post-attack network mechanism.
  </p>
  {main_figures}
  {robustness_figure}
  {quality_figure}

  <h2>5. Fixed-Effect Models And Robustness</h2>
  <p>
    The model table reports the condition-level alignment coefficient for the two planned outcomes. The robustness table
    adds HC3 standard errors, within-attack permutation p-values, and targeted exclusions for known upstream fallback or
    heuristic-warning rows. The permutation shuffles achieved alignment within attack family, preserving attack-level
    outcome structure while breaking the alignment-outcome pairing.
  </p>
  {_table_html(models, rows=20, columns=["outcome", "model_id", "estimate", "std_error", "p_value", "r_squared", "n", "df_resid"])}
  {_table_html(robustness, rows=30, columns=["outcome", "model_id", "estimate", "std_error", "p_value", "n", "excluded_rows", "excluded_conditions", "permutation_count"])}

  <h2>6. Original Fixed Assignment Versus Counterfactual Branch</h2>
  <p>
    The original production Run 2 is still scientifically useful, but it was not designed to create strong variation in
    susceptible profiles occupying influential sender positions. The table below is included only as context: the branch
    widens the alignment range intentionally and therefore provides the cleaner H3/H4 mechanism test.
  </p>
  {_table_html(original_comparison, rows=20)}

  <h2>7. Quality Gates And Condition Table</h2>
  {_table_html(quality, rows=10)}
  {_table_html(condition, rows=20, columns=["alignment_condition_id", "opinion_label", "attack_label", "target_alignment_z", "achieved_alignment_z", "alignment_abs_error", "n_profiles", "mean_pn_increment_effectivity", "mean_ae_total_network"])}
</body>
</html>
"""
    report_path.write_text(html_text, encoding="utf-8")
    return report_path
