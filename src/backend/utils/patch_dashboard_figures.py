"""
Patch the existing interactive_sem_dashboard.html to add/update static figure tabs
for the new violin plot, 3D moderation ISD, and fixed SEM diagram (no note).
Embeds PNGs as base64 <img> tags in new tab panels.
"""
from __future__ import annotations

import base64
import re
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[3]
DASHBOARD_PATH = (
    PROJECT_ROOT
    / "evaluation" / "run_1" / "stage_outputs"
    / "07_generate_research_visuals" / "interactive_sem_dashboard.html"
)
FIGURES_DIR = PROJECT_ROOT / "evaluation" / "run_1" / "publication_assets" / "figures"

# figures to inject as tabs: (tab_title, png_filename, caption_text)
STATIC_TABS = [
    (
        "Violin: Abs. Opinion Shift",
        "figure_2_absolute_delta_distribution.png",
        "Absolute attacked opinion shift by opinion leaf. "
        "Violin density, IQR bars, individual points (jitter), and mean diamonds. "
        "100 profiles × 4 attack vectors per leaf.",
    ),
    (
        "3D Moderation ISD",
        "figure_5_3d_susceptibility_isd.png",
        "Inter-individual moderation strength across the 4 × 4 attack–opinion factorial. "
        "Bar height = standard deviation of adversarial effectivity across 100 profiles. "
        "Higher bars = larger individual-difference spread in that attack–opinion cell.",
    ),
    (
        "SEM Path Diagram",
        "figure_4_annotated_sem_path_diagram.png",
        "Repeated-outcome SEM path coefficients from profile moderators to attacked "
        "opinion-shift indicators. Stars = p < .05; daggers = p < .10.",
    ),
]


def _img_tag(png_path: Path, alt: str) -> str:
    data = base64.b64encode(png_path.read_bytes()).decode("ascii")
    return (
        f'<img src="data:image/png;base64,{data}" '
        f'alt="{alt}" '
        f'style="max-width:100%;height:auto;border-radius:8px;box-shadow:0 4px 16px rgba(0,0,0,0.10);" />'
    )


def _static_tab_panel(idx_offset: int, idx: int, title: str, img_tag: str, caption: str) -> str:
    tab_id = f"tab-{idx_offset + idx}"
    return (
        f"<section id='{tab_id}' class='tab-panel'>"
        f"<h2>{title}</h2>"
        f"<div style='text-align:center;padding:10px 0;'>"
        f"{img_tag}"
        f"<p style='font-size:0.88rem;color:#4a5d7a;margin-top:10px;max-width:820px;margin-left:auto;margin-right:auto;'>"
        f"{caption}"
        f"</p>"
        f"</div>"
        f"</section>"
    )


def _static_tab_btn(idx_offset: int, idx: int, title: str) -> str:
    tab_id = f"tab-{idx_offset + idx}"
    return f"<button class='tab-btn' data-tab='{tab_id}'>{title}</button>"


def patch(html: str) -> str:
    # Count existing tabs to determine offset
    existing_tabs = re.findall(r"data-tab='tab-(\d+)'", html)
    if existing_tabs:
        max_idx = max(int(x) for x in existing_tabs)
        idx_offset = max_idx + 1
    else:
        idx_offset = 10  # fallback

    # Build new buttons and panels
    new_btns = "\n".join(
        _static_tab_btn(idx_offset, i, title) for i, (title, _, _) in enumerate(STATIC_TABS)
    )
    new_panels = "\n".join(
        _static_tab_panel(
            idx_offset, i, title,
            _img_tag(FIGURES_DIR / png, title),
            caption,
        )
        for i, (title, png, caption) in enumerate(STATIC_TABS)
    )

    # Inject buttons into the nav div (before </div> that closes .nav)
    # Find the nav section
    nav_close = html.rfind("</div>", html.find("class='nav'"), html.find("class='tab-panel'"))
    if nav_close == -1:
        # fallback: append before first </section>
        html = html.replace("</section>", f"{new_btns}\n</section>", 1)
    else:
        html = html[:nav_close] + "\n" + new_btns + "\n" + html[nav_close:]

    # Inject panels before the notes div
    notes_pos = html.find("<div class='notes'>")
    if notes_pos == -1:
        html += "\n" + new_panels
    else:
        html = html[:notes_pos] + new_panels + "\n" + html[notes_pos:]

    return html


if __name__ == "__main__":
    original = DASHBOARD_PATH.read_text(encoding="utf-8")

    # Check if already patched
    if "figure_5_3d_susceptibility_isd.png" in original or "3D Moderation ISD" in original:
        print("Dashboard already patched for these figures; skipping.")
    else:
        patched = patch(original)
        DASHBOARD_PATH.write_text(patched, encoding="utf-8")
        size_kb = DASHBOARD_PATH.stat().st_size / 1024
        print(f"Dashboard patched. New size: {size_kb:.0f} KB at {DASHBOARD_PATH}")
