"""Run a single-variant local AlphaGenome demonstration.

    uv run python score_variants.py --check-only
    uv run python score_variants.py --output-dir demo_scores
    uv run python score_variants.py --chromosome chr9 --position 27573483 \
        --ref C --alt CGGGCC --output-dir insertion_scores

Positions are 1-based on GRCh38; the FASTA needs its .fai index. Without
--ref/--alt, a dummy SNV is built
using the actual FASTA reference base. This is a software check, not a known
disease association. Scoring needs an allocated GPU and existing Hugging Face
authentication (hf auth login or HF_TOKEN). No inference API key is used.
The TSV contains every returned gene/track row, not a candidate ranking.
RNA_SEQ quantiles are signed; ACTIVE scorers need not describe effect direction.
"""

import argparse
import dataclasses
from datetime import datetime, timezone
import hashlib
from importlib import metadata
import json
import logging
import os
from pathlib import Path

import fsspec
from alphagenome.data import genome
from alphagenome.io import fasta
from alphagenome.models.v1 import dna_client, variant_scorers
from alphagenome_research.model import dna_model
from alphagenome_research.model.variant_scoring.calibration import calibration


LOGGER = logging.getLogger(__name__)
CALIBRATION_URL = (
    "https://storage.googleapis.com/alphagenome/data/hg38/calibration_scores.pb"
)
DEFAULT_SCORERS = ["RNA_SEQ", "ATAC", "DNASE", "CHIP_HISTONE", "CHIP_TF"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--fasta", type=Path,
        default=Path(os.environ.get(
            "ALPHAGENOME_FASTA", "/data/Dcode/gaetano/repos/fasta_files/hg38.fa"
        )),
        help="Local GRCh38 FASTA; defaults to ALPHAGENOME_FASTA or the notebook path.",
    )
    parser.add_argument("--chromosome", default="chr9")
    parser.add_argument("--position", type=int, default=27573483, help="1-based position.")
    parser.add_argument("--ref", help="Reference allele; supply together with --alt.")
    parser.add_argument("--alt", help="Alternate allele; supply together with --ref.")
    parser.add_argument(
        "--scorers", nargs="+", default=DEFAULT_SCORERS,
        choices=sorted(variant_scorers.RECOMMENDED_VARIANT_SCORERS),
        help="Defaults to directional RNA-seq/accessibility/ChIP scorers.",
    )
    parser.add_argument("--output-dir", type=Path, default=Path("demo_scores"))
    parser.add_argument(
        "--check-only", action="store_true",
        help="Validate the variant and calibration without loading a model or scoring.",
    )
    return parser.parse_args()


def prepare_variant(
    fasta_path: Path, chromosome: str, position: int,
    reference: str | None, alternate: str | None,
) -> genome.Variant:
    if not fasta_path.is_file():
        raise FileNotFoundError(f"GRCh38 FASTA not found: {fasta_path}")
    if position < 1:
        raise ValueError("Position must be a positive, 1-based coordinate.")
    if (reference is None) != (alternate is None):
        raise ValueError("Supply both --ref and --alt, or neither for a dummy SNV.")
    if reference is not None and alternate is not None:
        reference = reference.upper()
        alternate = alternate.upper()
        if not reference or not alternate or set(reference + alternate) - set("ACGT"):
            raise ValueError("Alleles must be nonempty A/C/G/T sequences; anchor indels.")
    extractor = fasta.FastaExtractor(fasta_path)
    length = len(reference) if reference is not None else 1
    observed = extractor.extract(
        genome.Interval(chromosome, position - 1, position - 1 + length)
    ).upper()
    if len(observed) != length or set(observed) - set("ACGT"):
        raise ValueError("The reference interval is out of bounds or has ambiguous bases.")
    if reference is None:
        reference = observed
        alternate = next(base for base in "ACGT" if base != reference)
    elif reference != observed:
        raise ValueError(f"REF mismatch: supplied {reference}, FASTA contains {observed}.")
    if alternate is None:
        raise ValueError("An alternate allele is required.")
    if reference == alternate:
        raise ValueError("REF and ALT must differ.")
    return genome.Variant(chromosome, position, reference, alternate)


def main() -> None:
    args = parse_args()
    variant = prepare_variant(
        args.fasta, args.chromosome, args.position, args.ref, args.alt
    )
    interval = variant.reference_interval.resize(dna_client.SEQUENCE_LENGTH_1MB)
    LOGGER.info("%s variant: %s", "Dummy" if args.ref is None else "Input", variant)
    scorers = [variant_scorers.RECOMMENDED_VARIANT_SCORERS[n] for n in args.scorers]
    calibration_path = fsspec.open_local(
        f"simplecache::{CALIBRATION_URL}", mode="rb",
        simplecache={"cache_storage": str(Path.home() / ".cache" / "alphagenome" / "calibration")},
        https={"client_kwargs": {"trust_env": True}},
    )
    calibrator = calibration.load(calibration_path)
    missing = [s.name for s in scorers if not calibrator.has_variant_scorer(s.name)]
    if missing:
        raise ValueError(f"Official human calibration is unavailable for: {missing}")
    report_path = args.output_dir / ("preflight.json" if args.check_only else "run.json")
    scores_path = args.output_dir / "scores.tsv"
    if report_path.exists() or (not args.check_only and scores_path.exists()):
        raise FileExistsError("Output exists; choose a new --output-dir to avoid overwriting.")
    report = {
        "status": "checked_only" if args.check_only else "completed",
        "started_at_utc": datetime.now(timezone.utc).isoformat(),
        "dummy_variant": args.ref is None,
        "variant": dataclasses.asdict(variant),
        "interval": str(interval),
        "genome_build": "GRCh38",
        "fasta": str(args.fasta.resolve()),
        "model_version": "fold-1",
        "scorers": {name: scorer.name for name, scorer in zip(args.scorers, scorers)},
        "calibration_url": CALIBRATION_URL,
        "calibration_sha256": hashlib.sha256(Path(calibration_path).read_bytes()).hexdigest(),
        "versions": {name: metadata.version(name) for name in (
            "alphagenome", "alphagenome-research", "jax", "jaxlib"
        )},
        "research_source": json.loads(
            metadata.distribution("alphagenome-research").read_text("direct_url.json") or "null"
        ),
    }
    if not args.check_only:
        import jax
        from huggingface_hub import get_token

        gpu_devices = [device for device in jax.local_devices() if device.platform == "gpu"]
        if not gpu_devices:
            raise RuntimeError("No JAX GPU available. Run on an allocated Biowulf GPU node.")
        if get_token() is None:
            raise RuntimeError("Authenticate first with `hf auth login` or set HF_TOKEN.")
        human = dna_client.Organism.HOMO_SAPIENS
        organism_settings = {
            human: dataclasses.replace(
                dna_model.default_organism_settings()[human],
                fasta_path=str(args.fasta.resolve()), calibration_path=calibration_path,
            ),
            dna_client.Organism.MUS_MUSCULUS: dna_model.OrganismSettings(),
        }
        LOGGER.info("Loading fold-1 on %s", gpu_devices[0])
        model = dna_model.create_from_huggingface(
            "fold-1", organism_settings=organism_settings, device=gpu_devices[0]
        )
        scores = model.score_variant(interval=interval, variant=variant, variant_scorers=scorers)
        if not scores or any("quantiles" not in score.layers for score in scores):
            raise RuntimeError("Model returned no scores or omitted required quantile calibration.")
        table = variant_scorers.tidy_scores([scores], match_gene_strand=True)
        if table is None or table.empty or "quantile_score" not in table:
            raise RuntimeError("No usable score table with quantile_score was returned.")
        report["score_rows"] = len(table)
        report["device"] = str(gpu_devices[0])
        args.output_dir.mkdir(parents=True, exist_ok=True)
        table.to_csv(scores_path, sep="\t", index=False)
        LOGGER.info("Wrote %s rows to %s", len(table), scores_path)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2) + "\n")
    LOGGER.info("Wrote %s (inference %s)", report_path, "not run" if args.check_only else "completed")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s", force=True)
    main()
