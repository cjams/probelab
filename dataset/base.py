import random

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Iterator, Callable
from enum import Enum
from pathlib import Path

def instructionify_nop(text: str):
    return text

def instructionify_activity(activity: str):
    return f"Give me instructions for {activity}"

class ExampleType(str, Enum):
    COMMAND = "command"
    ACTIVITY = "activity"
    STATEMENT = "statement"

@dataclass
class Example:
    id: int
    text: str
    label: bool
    type: ExampleType
    metadata: dict[str, Any] = field(default_factory=dict)
    instructionify: Callable[[str], str] | None = None

class ProbeDataset:
    def __init__(
        self,
        examples: list[Example],
        concept: str,
        pos_label: str,
        neg_label: str,
        source_datasets: set[str] | None = None,
    ):
        self.examples = examples
        self.concept = concept
        self.pos_label = pos_label
        self.neg_label = neg_label
        
        if source_datasets is not None:
            self.source_datasets = source_datasets
        else:
            self.source_datasets = {
                ex.metadata["source"]
                for ex in examples
                if "source" in ex.metadata
            }

    def __len__(self) -> int:
        return len(self.examples)

    def __iter__(self) -> Iterator[Example]:
        return iter(self.examples)

    def __getitem__(self, key: int | slice) -> "Example | ProbeDataset":
        if isinstance(key, slice):
            return ProbeDataset(self.examples[key], self.concept, self.pos_label, self.neg_label)

        return self.examples[key]

    def positives(self) -> "ProbeDataset":
        return ProbeDataset(
            [ex for ex in self.examples if ex.label],
            self.concept, self.pos_label, self.neg_label,
        )

    def negatives(self) -> "ProbeDataset":
        return ProbeDataset(
            [ex for ex in self.examples if not ex.label],
            self.concept, self.pos_label, self.neg_label,
        )

    def join(self, other: "ProbeDataset") -> "ProbeDataset":
        """Concatenate two compatible ProbeDatasets (same concept and labels)."""
        if self.concept != other.concept:
            raise ValueError(
                f"Cannot join datasets with different concepts: "
                f"'{self.concept}' vs '{other.concept}'"
            )

        if self.pos_label != other.pos_label or self.neg_label != other.neg_label:
            raise ValueError(
                f"Cannot join datasets with different labels: "
                f"('{self.pos_label}'/'{self.neg_label}') vs ('{other.pos_label}'/'{other.neg_label}')"
            )

        return ProbeDataset(
            self.examples + other.examples,
            self.concept, self.pos_label, self.neg_label,
            source_datasets=self.source_datasets | other.source_datasets,
        )

    def split(
        self,
        left_ratio: float = 0.8,
        seed: int = 42
    ) -> tuple["ProbeDataset", "ProbeDataset"]:
        """Split one dataset into two datasets with balanced classes in each split."""
        rng = random.Random(seed)

        def split_group(examples: list[Example]) -> tuple[list[Example], list[Example]]:
            shuffled = examples.copy()
            rng.shuffle(shuffled)
            n_left = round(len(shuffled) * left_ratio)

            return shuffled[:n_left], shuffled[n_left:]

        pos_left, pos_right = split_group(self.positives().examples)
        neg_left, neg_right = split_group(self.negatives().examples)

        def make(examples: list[Example]) -> "ProbeDataset":
            return ProbeDataset(examples, self.concept, self.pos_label, self.neg_label)

        return make(pos_left + neg_left), make(pos_right + neg_right)

    # Returns subset of examples where fn(example) is True
    def filter(self, fn: Callable[[Example], bool]) -> "ProbeDataset":
        return ProbeDataset(
            [ex for ex in self.examples if fn(ex)],
            self.concept, self.pos_label, self.neg_label,
        )

    def filter_by_type(self, type: ExampleType) -> "ProbeDataset":
        return self.filter(lambda ex: ex.type == type)

    def balance(self, seed: int | None = None) -> "ProbeDataset":
        """Return a new ProbeDataset with equal positive and negative examples (min of the two)."""
        rng = random.Random(seed)
        pos = self.positives().examples
        neg = self.negatives().examples
        n = min(len(pos), len(neg))
        sampled = rng.sample(pos, n) + rng.sample(neg, n)

        return ProbeDataset(sampled, self.concept, self.pos_label, self.neg_label)

    def label_balance(self) -> dict[str, int]:
        return {
            self.pos_label: len(self.positives()),
            self.neg_label: len(self.negatives())
        }

    def type_distribution(self) -> dict[ExampleType, int]:
        distribution: dict[ExampleType, int] = {}

        for ex in self.examples:
            distribution[ex.type] = distribution.get(ex.type, 0) + 1

        return distribution

    def save(self, path: Path) -> None: ...

    @classmethod
    def load(cls, path: Path) -> "ProbeDataset": ...

class DatasetLoader(ABC):
    """
    Abstract base for all dataset loaders.

    Subclasses must implement:
        - concept, pos_label, neg_label as properties, so the loader
          self-describes before load() is called
        - example_type, since a given loader produces a homogeneous
          example type
        - _load() - the actual loading

    Callers should call load_and_validate() rather than _load() directly.
    """

    @property
    @abstractmethod
    def concept(self) -> str:
        """Human-readable name for the binary concept, e.g. 'harmful_command'."""
        ...

    @property
    @abstractmethod
    def pos_label(self) -> str:
        """Name for the positive class, e.g. 'harmful'."""
        ...

    @property
    @abstractmethod
    def neg_label(self) -> str:
        """Name for the negative class, e.g. 'benign'."""
        ...

    @property
    @abstractmethod
    def example_type(self) -> ExampleType:
        """
        The ExampleType produced by this loader.
        A loader that produces a single type of example makes downstream
        prompt design and token selection simpler to reason about.
        """
        ...

    @abstractmethod
    def _load(self) -> ProbeDataset:
        """Load and return a ProbeDataset. Called by load()."""
        ...

    def validate(self, dataset: ProbeDataset) -> None:
        """
        Validate a loaded dataset. Raises ValueError on any issue.
        Subclasses can override to add loader-specific checks.
        """
        if len(dataset) == 0:
            raise ValueError(f"{self.__class__.__name__} loaded 0 examples.")

        if dataset.concept != self.concept:
            raise ValueError(
                f"Loaded dataset concept '{dataset.concept}' does not match "
                f"loader concept '{self.concept}'."
            )

        if dataset.pos_label != self.pos_label or dataset.neg_label != self.neg_label:
            raise ValueError(
                f"Loaded dataset labels ('{dataset.pos_label}'/'{dataset.neg_label}') "
                f"do not match loader labels ('{self.pos_label}'/'{self.neg_label}')."
            )

        wrong_type = [
            ex.id for ex in dataset
            if ex.type != self.example_type
        ]

        if wrong_type:
            raise ValueError(
                f"{len(wrong_type)} examples have a type other than "
                f"'{self.example_type}': {wrong_type[:5]}{'...' if len(wrong_type) > 5 else ''}"
            )

    def load(self) -> ProbeDataset:
        """Load, validate, and return the dataset."""
        dataset = self._load()
        self.validate(dataset)

        return dataset
