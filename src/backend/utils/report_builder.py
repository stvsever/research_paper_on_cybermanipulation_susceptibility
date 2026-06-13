from __future__ import annotations

import json
import math
import re
import subprocess
from pathlib import Path
from typing import Any, Dict, Iterable, List

import pandas as pd

from src.backend.utils.io import abs_path, ensure_dir, write_json, write_text


def _fmt(value: Any, digits: int = 3) -> str:
    if value is None:
        return "n/a"
    try:
        number = float(value)
        if math.isnan(number):
            return "n/a"
        return f"{number:.{digits}f}"
    except Exception:
        return str(value)


def _latex_escape(value: Any) -> str:
    text = str(value)
    replacements = {
        "\\": "\\textbackslash{}",
        "_": "\\_",
        "&": "\\&",
        "%": "\\%",
        "#": "\\#",
        "$": "\\$",
    }
    for old, new in replacements.items():
        text = text.replace(old, new)
    return text


def _render_bib() -> str:
    return r"""
@article{benjamini1995controlling,
  title = {Controlling the False Discovery Rate: A Practical and Powerful Approach to Multiple Testing},
  author = {Benjamini, Yoav and Hochberg, Yosef},
  journal = {Journal of the Royal Statistical Society: Series B (Methodological)},
  volume = {57},
  number = {1},
  pages = {289--300},
  year = {1995},
  doi = {10.1111/j.2517-6161.1995.tb02031.x}
}

@article{bennett2018disinformation,
  title = {The Disinformation Order: Disruptive Communication and the Decline of Democratic Institutions},
  author = {Bennett, W. Lance and Livingston, Steven},
  journal = {European Journal of Communication},
  volume = {33},
  number = {2},
  pages = {122--139},
  year = {2018},
  doi = {10.1177/0267323118760317}
}

@article{cioffirevilla2002invariance,
  title = {Invariance and Universality in Social Agent-Based Simulations},
  author = {Cioffi-Revilla, Claudio},
  journal = {Proceedings of the National Academy of Sciences},
  volume = {99},
  number = {suppl_3},
  pages = {7314--7316},
  year = {2002},
  doi = {10.1073/pnas.082081499}
}

@article{cameron2008bootstrap,
  title = {Bootstrap-Based Improvements for Inference with Clustered Errors},
  author = {Cameron, A. Colin and Gelbach, Jonah B. and Miller, Douglas L.},
  journal = {The Review of Economics and Statistics},
  volume = {90},
  number = {3},
  pages = {414--427},
  year = {2008},
  doi = {10.1162/rest.90.3.414}
}

@article{hung2022cognitive,
  title = {How China's Cognitive Warfare Works: A Frontline Perspective of Taiwan's Anti-Disinformation Wars},
  author = {Hung, Tzu-Chieh and Hung, Tzu-Wei},
  journal = {Journal of Global Security Studies},
  volume = {7},
  number = {4},
  pages = {ogac016},
  year = {2022},
  doi = {10.1093/jogss/ogac016}
}

@article{kozyreva2020citizens,
  title = {Citizens Versus the Internet: Confronting Digital Challenges with Cognitive Tools},
  author = {Kozyreva, Anastasia and Lewandowsky, Stephan and Hertwig, Ralph},
  journal = {Psychological Science in the Public Interest},
  volume = {21},
  number = {3},
  pages = {103--156},
  year = {2020},
  doi = {10.1177/1529100620946707}
}

@article{lazer2018science,
  title = {The Science of Fake News},
  author = {Lazer, David M. J. and Baum, Matthew A. and Benkler, Yochai and Berinsky, Adam J. and Greenhill, Kelly M. and Menczer, Filippo and Metzger, Miriam J. and Nyhan, Brendan and Pennycook, Gordon and Rothschild, David and Schudson, Michael and Sloman, Steven A. and Sunstein, Cass R. and Thorson, Emily A. and Watts, Duncan J. and Zittrain, Jonathan L.},
  journal = {Science},
  volume = {359},
  number = {6380},
  pages = {1094--1096},
  year = {2018},
  doi = {10.1126/science.aao2998}
}

@article{lewandowsky2017posttruth,
  title = {Beyond Misinformation: Understanding and Coping with the Post-Truth Era},
  author = {Lewandowsky, Stephan and Ecker, Ullrich K. H. and Cook, John},
  journal = {Journal of Applied Research in Memory and Cognition},
  volume = {6},
  number = {4},
  pages = {353--369},
  year = {2017},
  doi = {10.1016/j.jarmac.2017.07.008}
}

@article{matz2017psychological,
  title = {Psychological Targeting as an Effective Approach to Digital Mass Persuasion},
  author = {Matz, Sandra C. and Kosinski, Michal and Nave, Gideon and Stillwell, David J.},
  journal = {Proceedings of the National Academy of Sciences},
  volume = {114},
  number = {48},
  pages = {12714--12719},
  year = {2017},
  doi = {10.1073/pnas.1710966114}
}

@article{pennycook2019lazy,
  title = {Lazy, Not Biased: Susceptibility to Partisan Fake News Is Better Explained by Lack of Reasoning Than by Motivated Reasoning},
  author = {Pennycook, Gordon and Rand, David G.},
  journal = {Cognition},
  volume = {188},
  pages = {39--50},
  year = {2019},
  doi = {10.1016/j.cognition.2018.06.011}
}

@article{pennycook2021psychology,
  title = {The Psychology of Fake News},
  author = {Pennycook, Gordon and Rand, David G.},
  journal = {Trends in Cognitive Sciences},
  volume = {25},
  number = {5},
  pages = {388--402},
  year = {2021},
  doi = {10.1016/j.tics.2021.02.007}
}

@article{roozenbeek2022psychological,
  title = {Psychological Inoculation Improves Resilience Against Misinformation on Social Media},
  author = {Roozenbeek, Jon and van der Linden, Sander and Goldberg, B. and Rathje, Steve and Lewandowsky, Stephan},
  journal = {Science Advances},
  volume = {8},
  number = {34},
  pages = {eabo6254},
  year = {2022},
  doi = {10.1126/sciadv.abo6254}
}

@article{rosseel2012lavaan,
  title = {lavaan: An {R} Package for Structural Equation Modeling},
  author = {Rosseel, Yves},
  journal = {Journal of Statistical Software},
  volume = {48},
  number = {2},
  pages = {1--36},
  year = {2012},
  doi = {10.18637/jss.v048.i02}
}

@article{shrout1979intraclass,
  title = {Intraclass Correlations: Uses in Assessing Rater Reliability},
  author = {Shrout, Patrick E. and Fleiss, Joseph L.},
  journal = {Psychological Bulletin},
  volume = {86},
  number = {2},
  pages = {420--428},
  year = {1979},
  doi = {10.1037/0033-2909.86.2.420}
}

@article{vosoughi2018false,
  title = {The Spread of True and False News Online},
  author = {Vosoughi, Soroush and Roy, Deb and Aral, Sinan},
  journal = {Science},
  volume = {359},
  number = {6380},
  pages = {1146--1151},
  year = {2018},
  doi = {10.1126/science.aap9559}
}
""".strip() + "\n"


def _render_references_apa() -> str:
    references = [
        "Johns Hopkins University & Imperial College London. (2021, May 20). Countering cognitive warfare: Awareness and resilience. NATO Review. https://www.nato.int/docu/review/articles/2021/05/20/countering-cognitive-warfare-awareness-and-resilience/index.html",
        "Bennett, W. L., & Livingston, S. (2018). The disinformation order: Disruptive communication and the decline of democratic institutions. European Journal of Communication, 33(2), 122-139. https://doi.org/10.1177/0267323118760317",
        "Cioffi-Revilla, C. (2002). Invariance and universality in social agent-based simulations. Proceedings of the National Academy of Sciences, 99(Suppl. 3), 7314-7316. https://doi.org/10.1073/pnas.082081499",
        "Efron, B., & Tibshirani, R. J. (1993). An introduction to the bootstrap. Chapman & Hall/CRC.",
        "Hung, T.-C., & Hung, T.-W. (2022). How China's cognitive warfare works: A frontline perspective of Taiwan's anti-disinformation wars. Journal of Global Security Studies, 7(4), ogac016. https://doi.org/10.1093/jogss/ogac016",
        "Kozyreva, A., Lewandowsky, S., & Hertwig, R. (2020). Citizens versus the internet: Confronting digital challenges with cognitive tools. Psychological Science in the Public Interest, 21(3), 103-156. https://doi.org/10.1177/1529100620946707",
        "Lazer, D. M. J., Baum, M. A., Benkler, Y., Berinsky, A. J., Greenhill, K. M., Menczer, F., Metzger, M. J., Nyhan, B., Pennycook, G., Rothschild, D., Schudson, M., Sloman, S. A., Sunstein, C. R., Thorson, E. A., Watts, D. J., & Zittrain, J. L. (2018). The science of fake news. Science, 359(6380), 1094-1096. https://doi.org/10.1126/science.aao2998",
        "Lewandowsky, S., Ecker, U. K. H., & Cook, J. (2017). Beyond misinformation: Understanding and coping with the post-truth era. Journal of Applied Research in Memory and Cognition, 6(4), 353-369. https://doi.org/10.1016/j.jarmac.2017.07.008",
        "Matz, S. C., Kosinski, M., Nave, G., & Stillwell, D. J. (2017). Psychological targeting as an effective approach to digital mass persuasion. Proceedings of the National Academy of Sciences, 114(48), 12714-12719. https://doi.org/10.1073/pnas.1710966114",
        "Paulauskas, K. (2024, February 6). Why cognitive superiority is an imperative. NATO Review. https://www.nato.int/docu/review/articles/2024/02/06/why-cognitive-superiority-is-an-imperative/index.html",
        "Pennycook, G., & Rand, D. G. (2019). Lazy, not biased: Susceptibility to partisan fake news is better explained by lack of reasoning than by motivated reasoning. Cognition, 188, 39-50. https://doi.org/10.1016/j.cognition.2018.06.011",
        "Pennycook, G., & Rand, D. G. (2021). The psychology of fake news. Trends in Cognitive Sciences, 25(5), 388-402. https://doi.org/10.1016/j.tics.2021.02.007",
        "Roozenbeek, J., van der Linden, S., Goldberg, B., Rathje, S., & Lewandowsky, S. (2022). Psychological inoculation improves resilience against misinformation on social media. Science Advances, 8(34), eabo6254. https://doi.org/10.1126/sciadv.abo6254",
        "Rosseel, Y. (2012). lavaan: An R package for structural equation modeling. Journal of Statistical Software, 48(2), 1-36. https://doi.org/10.18637/jss.v048.i02",
        "Vosoughi, S., Roy, D., & Aral, S. (2018). The spread of true and false news online. Science, 359(6380), 1146-1151. https://doi.org/10.1126/science.aap9559",
    ]

    def _format_reference(ref: str) -> str:
        parts = re.split(r"(https?://\S+)", ref)
        rendered: List[str] = []
        for part in parts:
            if not part:
                continue
            if re.fullmatch(r"https?://\S+", part):
                rendered.append(rf"\url{{{part}}}")
            else:
                rendered.append(_latex_escape(part))
        return "".join(rendered)

    return "\n\n".join([f"\\noindent {_format_reference(ref)}\\par" for ref in references])


_ASSETS_FIGURES_ROOT: Path | None = None
_ASSETS_TABLES_ROOT: Path | None = None


def _table_input(filename: str) -> str:
    if _ASSETS_TABLES_ROOT is not None and not (_ASSETS_TABLES_ROOT / filename).exists():
        return rf"% table '{filename}' skipped: asset not generated for this run"
    return rf"\input{{../assets/tables/{filename}}}"


def _figure_block(filename: str, caption: str, label: str, note: str, width: str = "0.96\\linewidth") -> str:
    # Skip figures whose asset file was not produced by stage 08 so that the
    # LaTeX compile never halts on a missing include.
    if _ASSETS_FIGURES_ROOT is not None and not (_ASSETS_FIGURES_ROOT / filename).exists():
        return rf"% figure '{filename}' skipped: asset not generated for this run"
    return rf"""
\begin{{figure}}[H]
\raggedright
\caption{{{caption}}}
\includegraphics[width={width}]{{{filename}}}
\label{{{label}}}
\caption*{{\raggedright \footnotesize Note. {_latex_escape(note)}}}
\end{{figure}}
""".strip()


def _top_weight_text(weight_df: pd.DataFrame, n: int = 4) -> str:
    if weight_df.empty:
        return "Moderator weights were not available."
    top = weight_df.head(n).to_dict(orient="records")
    clauses = [
        f"{row['moderator_label']} ({_fmt(row['normalized_weight_pct'], 1)}% of normalized moderator weight)"
        for row in top
    ]
    return "; ".join(clauses)


def _top_group_text(weight_df: pd.DataFrame, n: int = 3) -> str:
    if weight_df.empty:
        return ""
    grouped = (
        weight_df.groupby("ontology_group", as_index=False)["normalized_weight_pct"]
        .sum()
        .sort_values("normalized_weight_pct", ascending=False)
        .head(n)
    )
    return "; ".join(
        f"{row['ontology_group']} ({_fmt(row['normalized_weight_pct'], 1)}%)"
        for row in grouped.to_dict(orient="records")
    )


def _maybe_mean(series: pd.Series) -> float | None:
    series = series.dropna()
    if len(series) == 0:
        return None
    return float(series.mean())


def _pretty_term(term: str) -> str:
    label = term
    for prefix in ["profile_cont_", "profile_cat__profile_cat_", "profile_cat__", "profile_cat_"]:
        if label.startswith(prefix):
            label = label[len(prefix) :]
    label = label.replace("_z", "")
    label = label.replace("__", " ")
    label = label.replace("_", " ")
    return label.title()


def _pretty_leaf(label: str) -> str:
    return label.replace("abs_delta_indicator__", "").replace("_", " ").title()


def _profile_highlights(profile_index_df: pd.DataFrame, n: int = 3) -> str:
    if profile_index_df.empty:
        return "not available"
    rows = profile_index_df.head(n).to_dict(orient="records")
    return "; ".join(
        f"{row['profile_id']} ({_fmt(row['susceptibility_index_pct'], 1)}th percentile; mean |Δ| = {_fmt(row['mean_abs_delta_score'], 1)})"
        for row in rows
    )


def _safe_corr(df: pd.DataFrame, left: str, right: str) -> float | None:
    if left not in df.columns or right not in df.columns:
        return None
    series = df[[left, right]].dropna()
    if len(series) < 3:
        return None
    value = series[left].corr(series[right], method="spearman")
    if pd.isna(value):
        return None
    return float(value)


def _render_tex(
    paper_title: str,
    run_id: str,
    config: Dict[str, Any],
    long_df: pd.DataFrame,
    profile_df: pd.DataFrame,
    profile_index_df: pd.DataFrame,
    sem_result: Dict[str, Any],
    ols_params: pd.DataFrame,
    bootstrap_params: pd.DataFrame,
    exploratory_df: pd.DataFrame,
    weight_df: pd.DataFrame,
) -> str:
    indicator_columns = [
        column
        for column in profile_df.columns
        if column.startswith("abs_delta_indicator__") and not column.endswith("_z")
    ]
    fit_indices = sem_result.get("fit_indices", {})
    cfi = fit_indices.get("CFI")
    rmsea = fit_indices.get("RMSEA")
    mean_abs_delta = float(long_df["abs_delta_score"].mean())
    mean_signed_delta = float(long_df["delta_score"].mean())
    mean_realism = _maybe_mean(long_df["attack_realism_score"]) if "attack_realism_score" in long_df.columns else None
    mean_plausibility = _maybe_mean(long_df["post_plausibility_score"]) if "post_plausibility_score" in long_df.columns else None
    attack_leaf = config.get("attack_leaf", "")
    focus_domain = config.get("focus_opinion_domain") or ", ".join(sorted(long_df["opinion_domain"].dropna().unique().tolist()))
    top_weight_text = _top_weight_text(weight_df)
    top_group_text = _top_group_text(weight_df)
    top_profiles_text = _profile_highlights(profile_index_df)
    bootstrap_lookup = {row["term"]: row for row in bootstrap_params.to_dict(orient="records")}
    sort_column = "normalized_weight_pct" if "normalized_weight_pct" in exploratory_df.columns else "multivariate_p_value"
    leading_terms = exploratory_df.sort_values(sort_column, ascending=False if sort_column == "normalized_weight_pct" else True).head(5) if not exploratory_df.empty else pd.DataFrame()
    moderator_text = "Exploratory moderator contrasts were not available."
    if not leading_terms.empty:
        moderator_text = "; ".join(
            f"{row['moderator_label']} (controlled b = {_fmt(row['univariate_estimate'], 2)}, p = {_fmt(row['univariate_p_value'], 3)}, weight = {_fmt(row.get('normalized_weight_pct'), 1)}%)"
            for row in leading_terms.to_dict(orient="records")
        )

    sem_path_df = pd.DataFrame(sem_result.get("coefficients", []))
    sem_path_df = sem_path_df.loc[sem_path_df["op"].astype(str) == "~"].copy() if not sem_path_df.empty else pd.DataFrame()
    sem_path_df["estimate"] = pd.to_numeric(sem_path_df.get("estimate"), errors="coerce")
    sem_path_df["p_value"] = pd.to_numeric(sem_path_df.get("p_value"), errors="coerce")
    sem_path_text = "No path coefficients reached the testing-run reporting threshold."
    if not sem_path_df.empty:
        highlighted = sem_path_df.loc[
            sem_path_df["p_value"].notna()
            & ~sem_path_df["rhs"].astype(str).str.contains("Mean Baseline|Mean Exposure", case=False, regex=True)
        ].sort_values("p_value").head(6)
        if not highlighted.empty:
            sem_path_text = "; ".join(
                f"{row['rhs']} -> {_pretty_leaf(str(row['lhs']))} (b = {_fmt(row['estimate'], 2)}, p = {_fmt(row['p_value'], 3)})"
                for row in highlighted.to_dict(orient="records")
            )

    index_alignment = _safe_corr(profile_index_df, "susceptibility_index_pct", "observed_effectivity_pct")
    index_alignment_text = _fmt(index_alignment, 3) if index_alignment is not None else "n/a"

    model_rows: List[str] = []
    for row in ols_params.to_dict(orient="records"):
        if row["term"] == "Intercept":
            continue
        boot = bootstrap_lookup.get(row["term"], {})
        model_rows.append(
            f"\\noindent\\textbf{{{_latex_escape(_pretty_term(str(row['term'])))}}}: b = {_fmt(row['estimate'], 3)}, p = {_fmt(row['p_value'], 3)}, bootstrap 95\\% CI [{_fmt(boot.get('conf_low'))}, {_fmt(boot.get('conf_high'))}]\\par"
        )
    model_rows_block = "\n".join(model_rows)

    return rf"""
\documentclass[11pt]{{article}}
\usepackage[a4paper,margin=1in]{{geometry}}
\usepackage{{amsmath}}
\usepackage{{amssymb}}
\usepackage{{graphicx}}
\usepackage{{booktabs}}
\usepackage{{threeparttable}}
\usepackage{{longtable}}
\usepackage{{tabularx}}
\usepackage{{caption}}
\usepackage{{subcaption}}
\usepackage{{fancyhdr}}
\usepackage{{hyperref}}
\usepackage{{microtype}}
\usepackage{{ragged2e}}
\usepackage{{setspace}}
\usepackage{{float}}
\usepackage{{array}}
\usepackage{{xcolor}}
\graphicspath{{{{../assets/figures/}}}}
\setstretch{{1.12}}
\captionsetup{{justification=RaggedRight,singlelinecheck=false}}
\pagestyle{{fancy}}
\fancyhf{{}}
\fancyhead[L]{{Multi-agent Simulation of Susceptibility to Cyber-manipulation}}
\fancyhead[R]{{Pilot Report}}
\fancyfoot[C]{{\thepage}}
\setlength{{\headheight}}{{14pt}}
\setlength{{\emergencystretch}}{{3em}}
\hypersetup{{hidelinks}}

\begin{{document}}
\begin{{center}}
{{\LARGE\bfseries {paper_title}\par}}
\vspace{{0.8em}}
{{\large Stijn Van Severen$^{{1,*}}$ \quad Thomas De Schryver$^1$\par}}
\vspace{{0.35em}}
{{\normalsize $^1$ Ghent University\par}}
{{\normalsize $^*$ Corresponding author\par}}
\vspace{{0.35em}}
{{\normalsize March 23, 2026\par}}
\end{{center}}

\begin{{abstract}}
This testing run examines how inter-individual differences moderate the effectivity of cyber-manipulation on political opinions using an ontology-driven multi-agent simulation pipeline. Test run 1 uses an attacked-only profile-panel design in which one fixed misinformation leaf is applied to each pseudoprofile across multiple political-opinion leaves. Attack effectivity is defined as within-profile post-minus-baseline movement, with absolute shift $A_{{ik}} = |\text{{post}}_{{ik}} - \text{{baseline}}_{{ik}}|$ as the primary outcome so that issue-specific signed movements do not cancel out. The testing run generated {len(profile_df)} pseudoprofiles, {len(long_df)} attacked opinion rows, and {len(indicator_columns)} repeated attacked outcomes per profile. Mean absolute attacked shift was {_fmt(mean_abs_delta, 2)}, mean signed shift was {_fmt(mean_signed_delta, 2)}, mean attack realism was {_fmt(mean_realism, 2)}, and mean post-exposure plausibility was {_fmt(mean_plausibility, 2)}. The structural model is now a repeated-outcome path SEM in which multiple attacked opinion deltas are regressed on profile moderators simultaneously. To compute a profile-level susceptibility summary without inheriting multicollinearity from a small-sample OLS, a cross-validated ridge aggregation is fit after the run and converted into a post hoc empirical susceptibility index. The testing run should therefore be read as methodological validation and exploratory evidence rather than as claim-ready causal estimation.
\end{{abstract}}

\section{{Introduction}}
Political influence campaigns increasingly operate in the space between mass communication, personalized persuasion, and platform-native misinformation. Rather than relying on technical compromise alone, they often work through selective framing, pseudo-consensus, strategic ambiguity, distrust activation, and identity-congruent narrative design (Bennett \& Livingston, 2018; Lazer et al., 2018; Lewandowsky et al., 2017; Pennycook \& Rand, 2019, 2021; Vosoughi et al., 2018). This makes the key research problem not only whether content is false, but whether the cognitive conditions of judgement are being shaped in ways that weaken self-directed political belief formation.

That concern is closely related to recent discussions of cognitive warfare and cognitive superiority in security policy. NATO-oriented analyses increasingly frame the contested terrain as one in which human attention, interpretation, trust, and decision-making become operationally relevant dimensions of conflict and competition (Johns Hopkins University \& Imperial College London, 2021; Hung \& Hung, 2022; Paulauskas, 2024). The present study adopts \textit{{cognitive sovereignty}} as the substantive framing for whether political opinion formation remains self-directed under digital adversarial pressure.

The research question is straightforward: \textit{{How do inter-individual differences moderate the effectivity of cyber-manipulation on political opinions?}} The methodological challenge is less straightforward. A real-world dataset that cleanly aligns profile structure, attack-vector semantics, repeated political-opinion items, and post-exposure response states is difficult to obtain, ethically constrained, and often poorly matched to ontology-level modeling. This makes a transparent, auditable simulation architecture useful as a methodological instrument, provided the architecture is explicit enough to be criticized and improved (Cioffi-Revilla, 2002; Rosseel, 2012).

\section{{Materials and Methods}}
\subsection{{Ontology-Constrained Pilot Design}}
Test run 1 uses three hierarchical ontologies: PROFILE, ATTACK, and OPINION. The ontologies remain hierarchical in storage, but estimation is leaf-based. PROFILE combines continuous leaves (e.g., age and Big Five means) with categorical leaves (e.g., sex). ATTACK contributes one fixed testing run leaf, \texttt{{{_latex_escape(attack_leaf)}}}. OPINION contributes repeated leaf nodes from the focused domain { _latex_escape(focus_domain) }.

The design is attacked-only. Each pseudoprofile is crossed with multiple opinion leaves, so one fixed attack leaf is connected to multiple attacked deltas for the same individual profile. This architecture is more coherent with the moderation question than a treatment-versus-control setup when the research focus is heterogeneity of attack effectivity under exposure rather than incremental exposure effects relative to no exposure.

\subsection{{Baseline, Exposure, and Post-Exposure Assessment}}
For each profile $i$ and opinion leaf $k$, the pipeline elicits a baseline opinion and a post-exposure opinion on a signed integer scale from $-1000$ to $+1000$. Exposure texts are generated as realistic social-media-native manipulative messages, then audited for realism. Post-exposure responses are audited for coherence and boundedness. This means every analyzed row is not merely a model output, but a generated output that has passed a second review layer designed to detect implausible reversals, coarse rounding, and issue-incoherent response shifts.

\subsection{{Effectivity Outcome and Profile-Level Susceptibility}}
The signed attacked shift for profile $i$ on opinion leaf $k$ is
\begin{{equation}}
\Delta_{{ik}} = \text{{post}}_{{ik}} - \text{{baseline}}_{{ik}},
\end{{equation}}
and the primary effectivity outcome is the absolute attacked shift
\begin{{equation}}
A_{{ik}} = |\Delta_{{ik}}|.
\end{{equation}}
The profile-level observed effectivity summary is the mean attacked shift magnitude
\begin{{equation}}
E_i = \frac{{1}}{{K}} \sum_{{k=1}}^K A_{{ik}},
\end{{equation}}
where $K$ is the number of repeated opinion leaves for profile $i$.

The repeated-outcome SEM is a profile-level path model:
\begin{{equation}}
A_{{ik}} = \alpha_k + X_i\beta_k + C_i\gamma_k + \varepsilon_{{ik}},
\end{{equation}}
where each repeated attacked opinion outcome $A_{{ik}}$ is regressed on the same profile vector $X_i$ and control vector $C_i$, while residual covariance between attacked opinion leaves is estimated within the SEM.

The post hoc empirical susceptibility index is not an input variable. It is derived after estimation from a target-conditional regularized aggregation. For each configured attack-opinion task $t \in \mathcal{{T}}$, a ridge model is fit:
\begin{{equation}}
\hat{{A}}_{{it}} = \hat{{\beta}}_{{0t}} + \sum_j \hat{{\beta}}_{{jt}}^{{(\mathrm{{ridge}})}} X_{{ij}},
\end{{equation}}
and the conditional susceptibility score is then aggregated as
\begin{{equation}}
S_i^*(\mathcal{{T}}) = \operatorname{{PctRank}}\left(\sum_{{t \in \mathcal{{T}}}} w_t \hat{{A}}_{{it}}\right), \qquad
w_t \propto \frac{{n_t}}{{\operatorname{{CV\mbox{{--}}MSE}}_t}}.
\end{{equation}}
This index is therefore descriptive. It summarizes the model-implied profile propensity toward larger attack effectivity for the configured attack/opinion target set while shrinking unstable coefficient combinations.

\section{{Results}}
Test run 1 generated {len(profile_df)} pseudoprofiles and {len(long_df)} attacked opinion rows. The fixed attack leaf was linked to {len(indicator_columns)} repeated opinion indicators per profile. Mean absolute attacked shift was {_fmt(mean_abs_delta, 2)} and mean signed shift was {_fmt(mean_signed_delta, 2)}. Mean attack realism was {_fmt(mean_realism, 2)}, while mean post-exposure plausibility was {_fmt(mean_plausibility, 2)}.

The measurement-and-moderation structure is summarized in Figure~\ref{{fig:design}}. Figure~\ref{{fig:absdelta}} shows the distribution of absolute attacked deltas across repeated opinion leaves. Figure~\ref{{fig:forest}} shows the descriptive susceptibility weights that summarize the post hoc profile index, and Figure~\ref{{fig:sem}} shows the repeated-outcome path-SEM coefficient matrix. The strongest normalized moderator weights in test run 1 were { _latex_escape(top_weight_text) }. At the ontology-group level, the largest aggregated weight shares were { _latex_escape(top_group_text) if top_group_text else 'not available' }.

{_figure_block('figure_1_study_design.pdf', 'Test run 1 attacked-only profile-panel design.', 'fig:design', 'One fixed ATTACK leaf is connected to multiple attacked OPINION leaves for each PROFILE. Hierarchical ontologies are preserved upstream, while estimation is leaf-based and repeated within profile.')}

{_figure_block('figure_2_absolute_delta_distribution.pdf', 'Absolute attacked opinion shift by repeated opinion leaf.', 'fig:absdelta', 'Absolute shift is the primary effectivity outcome because a single attack leaf can generate signed movement in different directions across different opinion leaves.')}

{_figure_block('figure_3_profile_moderator_coefficient_forest.pdf', 'Descriptive susceptibility weights across profile moderators.', 'fig:forest', 'Bars show the normalized contribution of each profile moderator to the post hoc susceptibility index under the configured attack/opinion target set.')}

{_figure_block('figure_4_annotated_sem_path_diagram.pdf', 'Repeated-outcome path-SEM coefficients across attacked opinion leaves.', 'fig:sem', 'Cells show SEM path coefficients from profile moderators to attacked opinion-shift indicators. The fixed ATTACK leaf is constant by design and therefore represented in the study design rather than as a varying regressor.')}

{_table_input('table_1_testing_run_design_and_configuration.tex')}
{_table_input('table_2_attacked_effectivity_descriptive_statistics.tex')}
{_table_input('table_3_multivariate_profile_moderator_model.tex')}

The repeated-outcome path SEM converged with CFI = {_fmt(cfi, 3)} and RMSEA = {_fmt(rmsea, 3)}. Those fit indices should still be read cautiously because the model is relatively dense for a testing run of this size and several paths are being screened simultaneously. The clearest leaf-specific paths were { _latex_escape(sem_path_text) }. These p-values are unadjusted and should be interpreted as exploratory. The post hoc susceptibility ranking identified the highest-scoring profiles as { _latex_escape(top_profiles_text) }. The Spearman alignment between the post hoc susceptibility ranking and the observed effectivity percentile was {index_alignment_text}, which is useful descriptively but should not be over-interpreted as external validation. Controlled summary contrasts and ridge-derived weights together suggested that the most influential profile components in this testing run were { _latex_escape(moderator_text) }.

\section{{Discussion}}
Test run 1 improves the methodological coherence of the study in three ways. First, the design now directly answers the moderation question under attack rather than drifting into a treatment-versus-control estimand. Second, effectivity is treated as a repeated construct: the same attack leaf is linked to multiple attacked opinion outcomes per profile, and those outcomes remain visible in the SEM instead of being collapsed prematurely. Third, the personal susceptibility index is no longer injected a priori as if it were observed truth. Instead, it is computed post hoc from target-conditional regularized task models and can therefore be interpreted as a model-derived profile summary.

These changes matter because the research question is not simply whether misinformation can move opinions. The more specific question is which inter-individual differences are associated with larger movement once exposure occurs. In that sense, the attacked-only repeated-leaf design is a cleaner testing-run architecture for studying heterogeneity of cyber-manipulation effectivity. The fixed attack leaf reduces attack-side heterogeneity, the repeated opinion leaves reduce item idiosyncrasy, and the moderator-weight table makes the final susceptibility ranking decomposable rather than opaque.

The testing run still has clear limits. The agents are synthetic. The exposure remains one fixed attack leaf. The sample, while larger than earlier internal tests, is still modest for high-confidence inference. The path-SEM fit is encouraging but should not be mistaken for definitive construct validation in a near-saturated testing-run setting. The post hoc susceptibility index is descriptive because it is derived from fitted model weights rather than observed externally. The current results therefore validate the research workflow more than they establish external truth about real populations.

\section{{Conclusion}}
This testing run demonstrates that an ontology-driven multi-agent simulation can be structured to study heterogeneity in cyber-manipulation effectivity with substantially better methodological alignment than the earlier internal tests. The core move is to treat one fixed attack leaf as producing multiple attacked opinion-shift indicators per profile, estimate those repeated outcomes jointly, and derive the final susceptibility index only after the full profile moderation structure has been estimated. Test run 1 is therefore a stronger methodological answer to the stated research question, even though it remains exploratory and should be scaled further before substantive claims are made.

\clearpage
\appendix
\section{{Supplementary Materials}}

{_figure_block('supplementary_figure_s1_baseline_post_scatter.pdf', 'Baseline versus post-attack opinion scores.', 'fig:s1', 'The diagonal marks no change. Points are colored by opinion leaf so issue-specific movement remains visible without collapsing the political state space to a single item.')}

{_figure_block('supplementary_figure_s2_profile_effectivity_heatmap.pdf', 'Per-profile attack effectivity heatmap.', 'fig:s2', 'Profiles are ordered by the post hoc empirical susceptibility index. Cells show mean absolute attacked shift for each repeated opinion leaf.')}

{_figure_block('supplementary_figure_s3_susceptibility_distribution.pdf', 'Distribution of the post hoc empirical susceptibility index.', 'fig:s3', 'The susceptibility index is the percentile-ranked profile-only linear predictor from the target-conditional ridge aggregation of fitted attack-opinion task models. It is descriptive rather than independent evidence.')}

{_table_input('supplementary_table_s1_ontology_leaves_used.tex')}
{_table_input('supplementary_table_s2_moderator_comparison.tex')}
{_table_input('supplementary_table_s3_assumption_and_risk_register.tex')}
{_table_input('supplementary_table_s4_reproducibility_manifest.tex')}
{_table_input('supplementary_table_s5_sem_path_coefficients.tex')}

\clearpage
\section*{{References}}
\begingroup
\small
\setlength{{\parindent}}{{-1.2em}}
\setlength{{\leftskip}}{{1.2em}}
{_render_references_apa()}
\endgroup

\end{{document}}
"""


_CUSTOM_TEX_SENTINEL = "% CUSTOM_MAIN_TEX"


def build_research_report(
    sem_long_csv_path: str | Path,
    sem_result_json_path: str | Path,
    ols_params_csv_path: str | Path,
    bootstrap_params_csv_path: str | Path,
    exploratory_comparison_csv_path: str | Path,
    config_json_path: str | Path,
    report_root: str | Path,
    report_assets_root: str | Path,
    paper_title: str,
    run_id: str,
) -> Dict[str, str]:
    report_root = ensure_dir(report_root)
    ensure_dir(report_assets_root)

    global _ASSETS_FIGURES_ROOT, _ASSETS_TABLES_ROOT
    _ASSETS_FIGURES_ROOT = Path(report_assets_root) / "figures"
    _ASSETS_TABLES_ROOT = Path(report_assets_root) / "tables"

    long_df = pd.read_csv(sem_long_csv_path)
    sem_result = json.loads(Path(sem_result_json_path).read_text(encoding="utf-8"))
    ols_params = pd.read_csv(ols_params_csv_path)
    bootstrap_params = pd.read_csv(bootstrap_params_csv_path)
    exploratory_df = pd.read_csv(exploratory_comparison_csv_path)
    config = json.loads(Path(config_json_path).read_text(encoding="utf-8"))

    stage05_dir = Path(sem_long_csv_path).resolve().parent
    stage06_dir = Path(sem_result_json_path).resolve().parent
    profile_df = pd.read_csv(stage05_dir / "profile_sem_wide.csv")
    profile_index_df = pd.read_csv(stage06_dir / "profile_susceptibility_index.csv")
    weight_table_path = stage06_dir / "moderator_weight_table.csv"
    weight_df = pd.read_csv(weight_table_path) if weight_table_path.exists() else pd.DataFrame()

    tex_path = Path(report_root) / "main.tex"
    bib_path = Path(report_root) / "references.bib"
    pdf_path = Path(report_root) / "main.pdf"
    summary_path = Path(report_root) / "report_summary.json"

    tex_content = _render_tex(
        paper_title=paper_title,
        run_id=run_id,
        config=config,
        long_df=long_df,
        profile_df=profile_df,
        profile_index_df=profile_index_df,
        sem_result=sem_result,
        ols_params=ols_params,
        bootstrap_params=bootstrap_params,
        exploratory_df=exploratory_df,
        weight_df=weight_df,
    )
    # Preserve manually crafted main.tex when sentinel comment is present.
    existing = tex_path.read_text(encoding="utf-8") if tex_path.exists() else ""
    if not existing.startswith(_CUSTOM_TEX_SENTINEL):
        write_text(tex_path, tex_content)
    write_text(bib_path, _render_bib())

    summary_payload = {
        "run_id": run_id,
        "n_profiles": int(profile_df["profile_id"].nunique()),
        "n_attacked_rows": int(len(long_df)),
        "n_repeated_indicators": int(long_df["opinion_leaf"].nunique()),
        "mean_abs_delta": float(long_df["abs_delta_score"].mean()),
        "mean_signed_delta": float(long_df["delta_score"].mean()),
        "mean_attack_realism": _maybe_mean(long_df["attack_realism_score"]) if "attack_realism_score" in long_df.columns else None,
        "mean_post_plausibility": _maybe_mean(long_df["post_plausibility_score"]) if "post_plausibility_score" in long_df.columns else None,
        "cfi": sem_result.get("fit_indices", {}).get("CFI"),
        "rmsea": sem_result.get("fit_indices", {}).get("RMSEA"),
    }
    write_json(summary_path, summary_payload)

    compile_errors: List[str] = []
    try:
        subprocess.run(["tectonic", "main.tex"], cwd=report_root, check=True, capture_output=True, text=True)
    except Exception as exc:
        compile_errors.append(str(exc))
        try:
            subprocess.run(["pdflatex", "-interaction=nonstopmode", "main.tex"], cwd=report_root, check=True, capture_output=True, text=True)
        except Exception as second_exc:
            compile_errors.append(str(second_exc))

    if not pdf_path.exists() and compile_errors:
        raise RuntimeError("Report compilation failed: " + " | ".join(compile_errors))

    return {
        "tex_path": abs_path(tex_path),
        "bib_path": abs_path(bib_path),
        "pdf_path": abs_path(pdf_path),
        "summary_path": abs_path(summary_path),
    }
