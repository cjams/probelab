# datasets/loaders/legal_activities.py

import pandas as pd
from ..base import DatasetLoader, Example, ExampleType, ProbeDataset, instructionify_activity

# Schema from legal_illegal_activities 
#   Activity — the description of the activity ['activity']
#   Status — legal or illegal ['legal_status']

class LegalActivitiesLoader(DatasetLoader):
    """
    Loads legal/illegal activites
    """

    concept = "illegal_acitivty"
    pos_label = "illegal"
    neg_label = "legal"
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
                label=(True if row['legal_status'] == "illegal" else False),
                type=ExampleType.ACTIVITY,
                instructionify=instructionify_activity,
                metadata={
                    "source": "legal_illegal_activities"
                },
            ))

        return ProbeDataset(
            examples=examples,
            concept=self.concept,
            pos_label=self.pos_label,
            neg_label=self.neg_label,
        )