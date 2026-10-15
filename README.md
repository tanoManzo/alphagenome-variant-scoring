# Local AlphaGenome variant scoring

A small command-line demonstration for scoring a human variant with the local
[AlphaGenome Research](https://github.com/google-deepmind/alphagenome_research)
model. It writes per-track raw effects and official calibrated quantile scores,
alongside a run manifest. An interactive notebook is also provided.

**Current scope:** one variant per invocation, human GRCh38, the `fold-1`
checkpoint, and a 1,048,576 bp sequence interval. This is not yet a batch
VCF/CSV pipeline or a validated disease-variant prioritization method.

## Files

| File | Purpose |
| --- | --- |
| `score_variants.py` | Single-variant command-line runner and preflight check |
| `test_score_variants.py` | Unit tests for reference validation and output writing |
| `Local_AlphaGenome.ipynb` | Interactive model loading, biosample exploration, and scoring |
| `pyproject.toml`, `uv.lock` | Dependency configuration and reproducible environment |

## Requirements

- Linux and [uv](https://docs.astral.sh/uv/).
- Python 3.13 is the locally validated interpreter. Project metadata permits
  Python 3.11 and later, but the full workflow has not been tested on every version.
- An allocated NVIDIA GPU visible to JAX for inference. The script does not
  request an HPC allocation or silently fall back to CPU inference.
- A local GRCh38 FASTA with its matching `.fai` index. Chromosome names must match
  the FASTA, for example `chr9`.
- Access to the upstream model checkpoint on Hugging Face, including acceptance
  of any required access terms, and Hugging Face authentication.
- Network access for the first checkpoint, annotation, and calibration downloads.

The pinned environment uses `alphagenome==0.10.0` and
`alphagenome-research==0.3.0` from commit
`3eecf48e58d7babbbc53525a8a0cf29a179e20a6`. The public model imports use
`alphagenome.models.v1`. A build constraint pins `grpcio-tools==1.78.0` so the
research package generates bindings compatible with the Protobuf 6 runtime.
Triton is pinned to 3.2.0, which provides a CPython 3.13 Linux wheel.

## Installation

From the repository directory:

```bash
uv python install 3.13
uv sync --locked --python 3.13
```

Commands below use `uv run`, so manually activating the environment is optional.
To set the terminal prompt without removing installed packages:

```bash
uv venv --allow-existing --python .venv/bin/python \
  --prompt alphagenome_biowulf .venv
source .venv/bin/activate
```

Set the reference FASTA and authenticate:

```bash
export ALPHAGENOME_FASTA=/path/to/GRCh38.fa
uv run hf auth login
```

Alternatively, configure `HF_TOKEN` securely in your shell environment. Do not
paste tokens into source files, notebook cells, command-line arguments, or Git.
No hosted AlphaGenome inference API key is needed: inference runs locally.

If the FASTA is not indexed, create its index using an installed FASTA indexing
tool, for example `samtools faidx /path/to/GRCh38.fa`.

On Biowulf, run inference within an existing GPU allocation. Configure the site's
DTN HTTP/HTTPS proxy when required. The calibration downloader honors proxy
environment variables; other downloads must also be able to reach their sources.

## 1. Check the setup without inference

```bash
uv run python score_variants.py --check-only --output-dir demo_scores
```

This checks the FASTA allele, builds a dummy SNV, loads the official human
calibration, and checks calibration availability for the requested scorers.
It writes `demo_scores/preflight.json` with `status: "checked_only"`.

**It does not load the model, test GPU inference, verify checkpoint access, or
generate prediction scores.** It may download calibration data if not cached.

The default dummy position is `chr9:27573483` (1-based). The reference base is
read from your FASTA; the alternate is the first different base in `ACGT`.
The dummy variant is a software example, not an asserted disease association.

## 2. Score the dummy variant

On a GPU node:

```bash
uv run python score_variants.py --output-dir demo_scores
```

An existing preflight report can remain in the directory. Existing `scores.tsv`
or `run.json` files are not overwritten: use a new output directory for another
inference run.

The runner loads `fold-1`, preserves the default human gene and splice
annotations, applies the local FASTA override, and enables official human
quantile calibration. The script requires Hugging Face authentication and a
JAX-visible GPU before loading the checkpoint.

## 3. Score a specified variant

Supply both alleles, using 1-based GRCh38 coordinates:

```bash
uv run python score_variants.py \
  --chromosome chr9 \
  --position 27573483 \
  --ref C \
  --alt A \
  --output-dir specified_variant_scores
```

Use `--fasta /path/to/GRCh38.fa` to override `ALPHAGENOME_FASTA`.
Always set one of these explicitly on a new machine; the script retains a
site-specific fallback from the original notebook.

Insertions and deletions must use nonempty, anchored alleles. Symbolic VCF alleles
such as `<DEL>`, ambiguous bases, identical REF/ALT, and REF mismatches are
rejected. The runner does not perform liftover, left-normalization, or automatic
chromosome-name conversion.

## Scorers

Defaults are:

| Name | Predicted readout |
| --- | --- |
| `RNA_SEQ` | Gene-associated RNA-seq effect |
| `ATAC` | Chromatin accessibility effect |
| `DNASE` | DNase accessibility effect |
| `CHIP_HISTONE` | Histone ChIP-seq signal effect |
| `CHIP_TF` | Transcription-factor ChIP-seq signal effect |

Choose a subset or different available scorers:

```bash
uv run python score_variants.py \
  --scorers RNA_SEQ ATAC DNASE \
  --output-dir selected_scorer_scores
```

Use `--help` for the supported scorer names. Availability in the client does not
guarantee that a scorer has official human calibration; the script rejects
uncalibrated choices rather than fabricating quantiles.

The notebook's example includes `*_ACTIVE` scorers, whereas the command-line
defaults use directional accessibility/ChIP scorers. These are different
summaries: ACTIVE scores should not automatically be interpreted as signed
increases or decreases. Keep the scorer definition attached to every result.

## Outputs

### `scores.tsv`

A long-format table containing all returned gene/track combinations, including:

- Variant identifier and scored interval.
- Gene identifiers and annotations where applicable.
- Output type and exact variant-scorer definition.
- Track, strand, biosample, ontology, and assay metadata where available.
- `raw_score`: the scorer-specific predicted effect.
- `quantile_score`: the corresponding official reference-calibrated score.

Non-gene scorers can have blank gene fields. Multiple rows for a variant reflect
different tracks or genes; they are not independent biological replicates.
No disease-specific biosample filter, gene selection, significance threshold,
or candidate ranking is applied.

Inspect a selected gene without discarding the scorer and tissue context:

```python
import pandas as pd

scores = pd.read_csv("demo_scores/scores.tsv", sep="\t")
columns = [
    "variant_id", "gene_name", "biosample_name",
    "output_type", "variant_scorer", "raw_score", "quantile_score",
]
print(scores.loc[scores["gene_name"].eq("C9orf72"), columns].to_string(index=False))
```

An empty selection means there are no returned rows for that gene; it does not
establish absence of a biological effect.

### `run.json`

The completed-run manifest records:

- Variant alleles, whether the variant was generated as a dummy, and interval.
- GRCh38 reference path, checkpoint label, scorer definitions, and device.
- Software versions and installed research-package source information.
- Calibration URL and SHA-256 checksum.
- Start timestamp and number of score rows.

`preflight.json` records the corresponding setup information without claiming
completed inference. Inspect the status before interpreting a report.
Manifests include local paths and variant details; treat them as research
outputs, not files to publish automatically.

## Calibration and interpretation

The official human calibration file is downloaded once from Google's public
storage and cached under `~/.cache/alphagenome/calibration`. Subsequent runs
reuse that file.

- RNA-seq quantiles can be negative. Their scale differs from ordinary unsigned
  percentiles and from other scorers.
- Quantiles are not P values, confidence intervals, pathogenicity probabilities,
  or experimental validation.
- Raw score units and direction depend on the scorer. Do not compare arbitrary
  raw values across assays as if they shared a common measurement scale.
- Upstream calibration can randomly break ties, so exact quantiles for tied
  values need not be bit-for-bit identical across runs.
- Model predictions do not establish enhancer/silencer function, target-gene
  causality, or a disease mechanism.
- Agreement with another predictor is not automatically independent validation.
  Check training overlap and evaluate against appropriate independent assays.

## Interactive notebook

Open `Local_AlphaGenome.ipynb`, select this repository's `.venv` interpreter,
and run from the imports cell onward. Set `ALPHAGENOME_FASTA` before starting
the kernel and configure Hugging Face authentication.

Restart the kernel after package upgrades. Previously loaded model objects
retain their old implementation. The published notebook is output-free;
saved tables and execution results should remain local.

## Tests and validation status

```bash
uv run python -m unittest -v test_score_variants
```

The tests cover dummy REF extraction, explicit insertions, invalid inputs,
raw/quantile TSV preservation, manifest writing, and overwrite protection.
The scoring test uses a mocked model and does not establish GPU correctness.

The real-reference/calibration preflight and four unit tests have passed in the
development environment. Full checkpoint loading and GPU inference are not
covered by those checks; verify them on your allocated GPU before relying on
results. This is a demonstration, not a clinically validated tool.

## Troubleshooting

| Problem | Action |
| --- | --- |
| No JAX GPU available | Use a GPU allocation and check the CUDA/JAX setup. |
| Missing Hugging Face authentication | Run `uv run hf auth login` or configure `HF_TOKEN`. |
| FASTA or index missing | Supply a readable GRCh38 FASTA with its matching `.fai`. |
| REF mismatch | Check genome build, 1-based position, chromosome name, and allele normalization. |
| Missing calibration/quantiles | Use the pinned versions and a scorer with official human calibration; check network/proxy access. |
| Protobuf runtime mismatch | Use `uv sync --locked`; retain the build compiler constraint. If an old incompatible research wheel is cached, run `uv cache clean alphagenome-research` followed by `uv sync --locked --reinstall-package alphagenome-research`. |
| Output already exists | Choose a new output directory; results are not silently replaced. |

## Upstream resources and data handling

- [AlphaGenome Research source](https://github.com/google-deepmind/alphagenome_research)
- [AlphaGenome client source](https://github.com/google-deepmind/alphagenome)
- [Official documentation](https://www.alphagenomedocs.com/)

This project is a wrapper around upstream software, not an official DeepMind
release. Model weights, datasets, and dependencies retain their own licenses
and access/reuse terms. They are not redistributed here.

Do not commit credentials, reference genomes, model checkpoints, participant
data, or variant results. The standalone repository excludes generated outputs
and publishes the notebook without execution outputs.
