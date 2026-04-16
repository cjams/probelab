# datasets/loaders/tdc2023.py

from datasets import load_dataset
from ..base import DatasetLoader, Example, ExampleType, ProbeDataset, instructionify_nop

# All examples are harmful commands (positive class only).

TDC2023_HF_PATH = "walledai/TDC23-RedTeaming"

class TDC2023Loader(DatasetLoader):
    """
    Loads harmful behavior prompts from the 2023 Trojan Detection Challenge dataset.
    """

    concept = "harmful_command"
    pos_label = "harmful"
    neg_label = "benign"
    example_type = ExampleType.COMMAND

    def __init__(self):
        pass

    def _load(self) -> ProbeDataset:
        ds = load_dataset(TDC2023_HF_PATH)

        examples = []
        for i, row in enumerate(ds['train']):
            examples.append(Example(
                id=i,
                text=row["prompt"],
                label=True, # all are harmful commands
                type=ExampleType.COMMAND,
                instructionify=instructionify_nop,
                metadata={
                    "source": "walledai/TDC2023-RedTeaming",
                },
            ))

        return ProbeDataset(
            examples=examples,
            concept=self.concept,
            pos_label=self.pos_label,
            neg_label=self.neg_label,
        )