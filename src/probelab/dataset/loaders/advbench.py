# datasets/loaders/advbench.py

from datasets import load_dataset
from ..base import DatasetLoader, Example, ExampleType, ProbeDataset, instructionify_nop

# Schema from walledai/AdvBench
#   prompt  — the harmful instruction/request
#   target  — the desired harmful completion the attack aims to elicit.
#             This maps to target_completion, similar to contextual harmbench
#
# No category metadata — AdvBench is a flat list with no semantic labels.
#
# Like HarmBench, all examples are harmful (positive class only).
# The "target" column is preserved in metadata as target_completion — it can
# be useful for prompt design (e.g. as a prefill in differential prompts) but
# is not part of the probe input text by default.

ADVBENCH_HF_PATH = "walledai/AdvBench"

class AdvBenchLoader(DatasetLoader):
    """
    Loads harmful instruction prompts from AdvBench.
    """

    concept      = "harmful_command"
    pos_label    = "harmful"
    neg_label    = "benign"
    example_type = ExampleType.COMMAND

    def __init__(self):
        pass

    def _load(self) -> ProbeDataset:
        ds = load_dataset(ADVBENCH_HF_PATH)

        examples = []
        for idx, row in enumerate(ds['train']):
            text = row["prompt"]

            examples.append(Example(
                id=idx,
                text=text,
                label=True, # all are harmful commands
                type=ExampleType.COMMAND,
                instructionify=instructionify_nop,
                metadata={
                    "source": "walledai/AdvBench",
                    # Always preserve target so PromptDesign can use it
                    # (e.g. as a prefill) even when it's not part of the probe input.
                    "target_completion": row["target"],
                }
            ))

        return ProbeDataset(
            examples=examples,
            concept=self.concept,
            pos_label=self.pos_label,
            neg_label=self.neg_label,
        )