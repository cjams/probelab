# datasets/loaders/harmbench.py

from datasets import load_dataset
from ..base import DatasetLoader, Example, ExampleType, ProbeDataset, instructionify_nop

# All examples are harmful commands (positive class only). Negatives must be supplied
# externally and combined via datasets.combine().
#
# category notes:
#   "standard"   — direct command, no context needed; the most commonly used subset
#   "contextual" — requires ContextString to be injected into the prompt. These are
#                  partial completions from a compliant model
#   "copyright"  — tests for IP reproduction; semantically different from harm
#
# For refusal probes, you almost certainly want functional_categories=["standard"].
# "contextual" is usable but requires your PromptDesign to handle ContextString.
# "copyright" is a different concept and should be a separate probe.

HARMBENCH_HF_PATH = "walledai/HarmBench"
FUNCTIONAL_CATEGORIES = {"standard", "contextual", "copyright"}

class HarmBenchLoader(DatasetLoader):
    """
    Loads harmful behavior prompts from HarmBench.

    Requires accepting the dataset terms at:
    https://huggingface.co/datasets/walledai/HarmBench

    Args:
        subset:
            Which subset to use.
            Defaults to ["standard"] — direct requests with no context needed.
            Pass ["standard", "contextual"] to include context-dependent behaviors,
            but note your PromptDesign will need to handle target_completion in that case.
    """

    concept = "harmful_command"
    pos_label = "harmful"
    neg_label = "benign"
    example_type = ExampleType.COMMAND

    def __init__(self, subset: str):
        self.subset = subset

    def _load(self) -> ProbeDataset:
        ds = load_dataset(HARMBENCH_HF_PATH, self.subset)

        examples = []
        for i, row in enumerate(ds['train']):
            examples.append(Example(
                id=i,
                text=row["prompt"],
                label=True, # all are harmful commands
                type=ExampleType.COMMAND,
                instructionify=instructionify_nop,
                metadata={
                    "source": "walledai/HarmBench",
                    "subset": self.subset,
                    "category": row['category'],
                    "target_completion": row["context"] if self.subset == "contextual" else ""
                },
            ))

        return ProbeDataset(
            examples=examples,
            concept=self.concept,
            pos_label=self.pos_label,
            neg_label=self.neg_label,
        )