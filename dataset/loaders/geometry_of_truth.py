# dataset/loaders/geometry_of_truth.py

import csv

from pathlib import Path

from ..base import DatasetLoader, Example, ExampleType, ProbeDataset, instructionify_nop

# Default data directory: probelab/data/. Overridable via data_dir=.
_DEFAULT_DATA_DIR = Path(__file__).resolve().parents[2] / "data"

# Datasets shipped under data/. Anything else is still loadable via path=.
KNOWN_DATASETS = (
    "cities",
    "companies_true_false",
    "larger_than",
    "sp_en_trans",
)


class GeometryOfTruthLoader(DatasetLoader):
    """
    Loads true/false statement datasets from Marks & Tegmark's
    "Geometry of Truth" repo (saprmarks/geometry-of-truth/datasets).

    Each CSV is a flat list with `statement,label` as the canonical first
    two columns; any additional columns — `city,country,correct_country`
    for cities, `n1,n2,diff,abs_diff` for larger_than, etc. — are preserved
    on Example.metadata so downstream filtering or analysis can use them.

    Args:
        name:     Dataset name (no extension), e.g. "cities" or
                  "sp_en_trans". Resolved to <data_dir>/<name>.csv.
        path:     Explicit path to a CSV. Overrides name when given —
                  useful for custom or out-of-repo datasets that follow
                  the same statement/label schema.
        data_dir: Directory containing the CSVs. Defaults to
                  <package>/data/.
    """

    concept      = "truth"
    pos_label    = "true"
    neg_label    = "false"
    example_type = ExampleType.STATEMENT

    def __init__(
        self,
        name: str | None = None,
        path: str | Path | None = None,
        data_dir: str | Path | None = None,
    ):
        if name is None and path is None:
            raise ValueError("GeometryOfTruthLoader requires either name= or path=.")

        self.name = name
        self.path = Path(path) if path is not None else None
        self.data_dir = Path(data_dir) if data_dir is not None else _DEFAULT_DATA_DIR

    def _resolve_path(self) -> Path:
        if self.path is not None:
            return self.path

        candidate = self.data_dir / f"{self.name}.csv"

        if not candidate.exists():
            raise FileNotFoundError(
                f"GeometryOfTruthLoader could not find {candidate!s}. "
                f"Place the CSV under {self.data_dir!s} or pass an explicit path=."
            )

        return candidate

    def _load(self) -> ProbeDataset:
        path = self._resolve_path()
        source = f"saprmarks/geometry-of-truth/{path.stem}"

        examples: list[Example] = []

        with path.open(newline="") as f:
            reader = csv.DictReader(f)

            for idx, row in enumerate(reader):
                examples.append(Example(
                    id=idx,
                    text=row["statement"],
                    label=bool(int(row["label"])),
                    type=ExampleType.STATEMENT,
                    instructionify=instructionify_nop,
                    metadata={
                        "source": source,
                        **{k: v for k, v in row.items() if k not in ("statement", "label")},
                    },
                ))

        return ProbeDataset(
            examples=examples,
            concept=self.concept,
            pos_label=self.pos_label,
            neg_label=self.neg_label,
        )
