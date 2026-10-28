# The wiring program of a fly brain — accompanying code

Companion repository for the manuscript *"The wiring program of a fly brain"*
(Xining Xun, Hang Fan, Tianyi Hao; submitted). It contains the complete,
numbered analysis pipeline behind every number reported in the paper and its
Supplementary Information, together with the manuscript build scripts.

## Contents

- `scripts/` — 192 numbered, single-purpose scripts covering data ingestion,
  module estimation, program generation, every registered falsification
  experiment, all controls and diagnostics, the statistical battery and figure
  production. Each script carries a header comment describing its function;
  section symbols (§) in the comments point to the timestamped project log,
  and every row of Supplementary Table 1 (the registered falsification ledger)
  traces back to specific scripts.
- `代码清单.md` — the script manifest (in Chinese), grouping all scripts into
  13 pipeline stages; one line per script.
- `paper_build/` — manuscript builders.
  - `build_paper.js` — builds the submission manuscript and the Supplementary
    Information (.docx) from `paper_doc.json` / `si_doc.json`:
    `npm install docx`, then `node build_paper.js out.docx paper_doc.json`.
  - `build_overleaf.py` — generates the LaTeX/Overleaf project
    (`main.tex`, `supplementary.tex`, `figures/`) from the same JSON sources.
  - `figs_submission/` — the seven manuscript figures.

## Data (not in this repository)

Both connectomes are publicly distributed by their respective projects:

- Male CNS: MaleCNS v1.0 flat connectome (164,506 typed neurons, 25,135,616
  connections), FlyEM — <https://storage.googleapis.com/flyem-male-cns/v1.0/>
- Female brain: FlyWire FAFB snapshot v783 via the Codex public bucket
  (138,327 typed neurons, 19.7 million connections) —
  <https://storage.googleapis.com/flywire-data/codex/data/fafb/783/>

Raw feather files and intermediate binary products are not redistributed
here (size and licensing); every intermediate product can be regenerated from
the two public sources with the scripts in this repository.

## Environment and reproduction

- Python 3 (numpy / pyarrow / scipy), bash; the large scripts are
  memory-mapped or chunked to run within a 4 GB memory budget.
- Node.js 20+ with the `docx` package, only for the manuscript builder.
- All randomized estimates use fixed seeds (wedge-sampler seeds 1–12;
  edge-split seed 11; lattice permutation seed 13). Generation and
  verification are deterministic-seed pipelines: every statistic quoted in the
  paper can be recomputed end-to-end by running the numbered scripts in order;
  each statistic is written to a machine-readable log as it is computed, and
  the values in the text were transcribed from those logs.
