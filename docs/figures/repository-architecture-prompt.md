# Repository architecture figure

Generated with the built-in ImageGen tool using the imagegen skill.

Figure: `repository-architecture.png`

Suggested caption: **Repository organization and layered architecture of LFTC.** The maintained Python package separates workflow orchestration, execution and reporting, reusable components, configuration, and shared domain types. Current assets and experiment outputs remain separate from preserved historical materials. The presentation reads historical results and saved checkpoints without importing archived Python code. Layer arrows denote the allowed dependency direction rather than a complete import graph.

## Final prompt

Use case: infographic-diagram.
Asset type: a publication-ready software repository architecture figure for an NLP research paper.
Primary request: accurately depict the CURRENT implemented LFTC text-classification repository described below. Produce ONE refined, high-resolution landscape figure, approximately 16:9, ideally 3840 x 2160. Clean vector-like engineering diagram, not a screenshot or decorative illustration. Pure white background. Thin consistent strokes, perfectly aligned rectangular boxes, generous spacing, crisp dark navy text. Restrained pale blue for maintained code, pale teal for current local assets, pale warm gray for preserved historical materials. Sans-serif labels, monospaced directory/package names. Strong readable typography suitable for a two-column paper spanning both columns. No gradients, shadows, 3D objects, mascots, code screenshots or random icons.

Title: "LFTC Repository Architecture"
Three panels with exact headings: "(a) Repository organization", "(b) Layered core package", "(c) Local assets and archive".
Panel widths approximately 29%, 43%, 28%.

PANEL (a): a precise, readable directory tree, not a flowchart. Root "sentiment-project/". Show these exact branches and their short secondary descriptions:
"src/lftc/" — "Active Python package"
"apps/" — "PPT-based presentation"
"configs/" — "YAML configurations"
"ops/" — "Launch, Slurm, packaging"
"tests/" — "Unit, integration, architecture"
"docs/" — "Usage, design, research"
"artifacts/" — "Assets and new outputs"
"archive/2026-11-08/" — "Preserved historical materials"
Below the tree, small text: "README.md   |   pyproject.toml".
Highlight src/lftc/ in blue, artifacts/ in teal, and archive/2026-11-08/ in warm gray. All other directory names dark navy.

PANEL (b): the active package "src/lftc/" with FIVE precisely aligned horizontal tiers. Tiers stack from L4 at top to L0 at bottom. The layer labels are on the left and the named package boxes are inside each row. Treat this as a LAYERING DIAGRAM, NOT a complete actual-import graph.
Top tier: "L4  Orchestration", with two boxes "cli" and "experiments". Under experiments, small text "prepare / run / predict / evaluate / report".
Second tier: "L3  Execution & reporting", with "training" and "reporting". Under training, small text "Training and saved-model inference"; under reporting, "Metrics, reports, presentation evidence". A second compact line under training says "benchmarks"; under reporting it says "presentation". These refer to the existing original-checkpoint loader and the PPT evidence reader.
Middle tier: "L2  Reusable components", with four equal boxes "data", "features", "models", "evaluation". Small descriptions respectively "Load & sample", "TF-IDF / Word2Vec", "Linear / neural", "Metrics & diagnostics".
Fourth tier: "L1  Configuration & paths", with two boxes "config", "governance".
Bottom tier: "L0  Shared domain", a single wide box "domain", small text "Tasks, assets, samples, shared types".
A single downward arrow alongside the five tiers, NOT multiple arrows among all boxes, labeled "Allowed dependency direction". This arrow encodes higher layers may use lower layers, not that every adjacent pair directly imports one another.
Below the layers in a narrow separate white strip: "Presentation uses reporting.presentation and training.benchmarks".
Do NOT place apps inside the Python package or assert the presentation uses experiments.

PANEL (c): two storage boxes, visually distinct from executable package tiers.
Upper teal box labeled "artifacts/" and subtitle "Current workspace assets and outputs". Include exactly the five rows "assets/   datasets and base models", "prepared/   processed datasets", "runs/   trained models and predictions", "results/   diagnostics", "reports/   exported reports". Show one thin teal bidirectional connector between this storage box and the core package, labeled "read / write". It represents the new workflow's local inputs and outputs, not an import dependency.
Lower warm gray box labeled "archive/2026-11-08/" with three simple rows "Historical result files", "Original model checkpoints", "Legacy source files". Small note INSIDE this box: "Legacy source is reference only".
One dashed blue connector from the archive box points LEFT into the L3 tier only, labeled "read results / weights". This is data access by the presentation evidence reader and model loader. Do NOT draw an arrow from archived source to active source; do not suggest old Python modules are imported.
At bottom of panel (c), a clear note: "No archived Python imports".

Small legend at bottom across the figure: black arrow "Layer dependency direction"; teal double arrow "Current asset I/O"; dashed blue arrow "Preserved evidence read".
All arrows must have clear endpoints, avoid crossing labels or boxes, and must obey the data access semantics above. No GitHub/cloud deployment or invented storage/services. Do not add packages or files. Ensure exact spelling of all path names including "governance", "experiments", "evaluation", "benchmarks", "presentation", "2026-11-08". Preserve generous whitespace; remove nonessential decoration rather than shrinking labels. Render all supplied labels accurately and legibly.

