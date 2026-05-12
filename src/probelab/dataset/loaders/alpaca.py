# datasets/loaders/alpaca.py

from datasets import load_dataset
from ..base import DatasetLoader, Example, ExampleType, ProbeDataset, instructionify_nop

# Schema from tatsu-lab/alpaca
# All examples are harmless commands (negative class only).

ALPACA_HF_PATH = "tatsu-lab/alpaca"

class AlpacaLoader(DatasetLoader):
    """
    Loads harmless behavior prompts from Alpaca.
    """

    concept = "harmful_command"
    pos_label = "harmful"
    neg_label = "benign"
    example_type = ExampleType.COMMAND

    def __init__(self):
        pass

    def _load(self) -> ProbeDataset:
        ds = load_dataset(ALPACA_HF_PATH)

        examples = []
        for i, row in enumerate(ds['train']):
            examples.append(Example(
                id=i,
                text=row["instruction"],
                label=False, # Alpaca is harmless commands only
                type=ExampleType.COMMAND,
                instructionify=instructionify_nop,
                metadata={
                    "source": "tatsu-lab/alpaca",
                    "target_completion": row["output"]
                }
            ))

        return ProbeDataset(
            examples=examples,
            concept=self.concept,
            pos_label=self.pos_label,
            neg_label=self.neg_label,
        )