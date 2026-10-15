import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import anndata
import numpy as np
import pandas as pd
import pyfaidx

import score_variants as scoring


class ScoringTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.fasta = self.root / "test.fa"
        self.fasta.write_text(">chr1\nACGTACGTACGT\n")
        pyfaidx.Faidx(str(self.fasta)).close()

    def test_dummy_variant_uses_reference(self) -> None:
        variant = scoring.prepare_variant(self.fasta, "chr1", 2, None, None)
        self.assertEqual(variant.reference_bases, "C")
        self.assertEqual(variant.alternate_bases, "A")

    def test_invalid_variants(self) -> None:
        for position, ref, alt in [
            (0, None, None), (2, "A", "G"), (2, "C", None),
            (2, "C", "C"), (2, "N", "A"), (50, None, None),
        ]:
            with self.subTest(position=position, ref=ref, alt=alt):
                with self.assertRaises(ValueError):
                    scoring.prepare_variant(self.fasta, "chr1", position, ref, alt)

    def test_explicit_insertion(self) -> None:
        variant = scoring.prepare_variant(self.fasta, "chr1", 2, "C", "CGG")
        self.assertEqual(variant.alternate_bases, "CGG")

    def test_scoring_output_preserves_scores(self) -> None:
        variant = scoring.prepare_variant(self.fasta, "chr1", 2, "C", "A")
        scorer = scoring.variant_scorers.RECOMMENDED_VARIANT_SCORERS["RNA_SEQ"]
        score = anndata.AnnData(
            X=np.array([[-0.25]], dtype=np.float32),
            obs=pd.DataFrame(
                {"gene_id": ["ENSG00000000001.1"], "gene_name": ["TEST"], "strand": ["+"]},
                index=["0"],
            ),
            var=pd.DataFrame(
                {"name": ["test_track"], "strand": ["+"], "ontology_curie": ["CL:TEST"]},
                index=["0"],
            ),
            uns={"variant": variant, "interval": variant.reference_interval,
                 "variant_scorer": scorer},
            layers={"quantiles": np.array([[-0.9]], dtype=np.float32)},
        )
        calibration_file = self.root / "calibration.pb"
        calibration_file.write_bytes(b"test fixture")
        output = self.root / "output"
        args = SimpleNamespace(
            fasta=self.fasta, chromosome="chr1", position=2, ref="C", alt="A",
            scorers=["RNA_SEQ"], output_dir=output, check_only=False,
        )
        with (
            patch.object(scoring, "parse_args", return_value=args),
            patch.object(scoring.fsspec, "open_local", return_value=str(calibration_file)),
            patch.object(scoring.calibration, "load") as load,
            patch("jax.local_devices", return_value=[SimpleNamespace(platform="gpu")]),
            patch("huggingface_hub.get_token", return_value="test-token"),
            patch.object(scoring.dna_model, "create_from_huggingface") as create,
        ):
            load.return_value.has_variant_scorer.return_value = True
            create.return_value.score_variant.return_value = [score]
            scoring.main()
            table = pd.read_csv(output / "scores.tsv", sep="\t")
            self.assertAlmostEqual(table.loc[0, "raw_score"], -0.25)
            self.assertAlmostEqual(table.loc[0, "quantile_score"], -0.9, places=6)
            report = json.loads((output / "run.json").read_text())
            self.assertEqual(report["status"], "completed")
            self.assertFalse(report["dummy_variant"])
            self.assertEqual(report["score_rows"], 1)
            self.assertNotIn("test-token", (output / "run.json").read_text())
            with self.assertRaises(FileExistsError):
                scoring.main()


if __name__ == "__main__":
    unittest.main()
