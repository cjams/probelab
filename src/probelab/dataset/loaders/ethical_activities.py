# datasets/loaders/ethical_activities.py

import pandas as pd
from ..base import DatasetLoader, Example, ExampleType, ProbeDataset, instructionify_activity

# Schema from ethical_unethical_actitivies
#   Activity — the description of the activity ['activity']
#   Status — ethical or unethical ['ethical_status']

class EthicalActivitiesLoader(DatasetLoader):
    """
    Loads ethical/unethical activites
    """

    concept = "unethical_acitivty"
    pos_label = "unethical"
    neg_label = "ethical"
    example_type = ExampleType.ACTIVITY

    def __init__(self, file_path):
        self.file_path = file_path

    def _load(self) -> ProbeDataset:
        df = pd.read_csv(self.file_path)

        examples = []
        for idx, (_, row) in enumerate(df.iterrows()):
            examples.append(Example(
                id=idx,
                text=row["activity"],
                label=(True if row['ethical_status'] == "unethical" else False),
                type=ExampleType.ACTIVITY,
                instructionify=instructionify_activity,
                metadata={
                    "source": "ethical_unethical_acitivities"
                },
            ))

        return ProbeDataset(
            examples=examples,
            concept=self.concept,
            pos_label=self.pos_label,
            neg_label=self.neg_label,
        )