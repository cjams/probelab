from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field


@dataclass
class SemanticJudgment:
    command: str
    response: str
    # 1 = response exhibits the property, 0 = does not, None = unclassified
    # (judge itself refused, or persistent provider/parse error).
    label: int | None


@dataclass
class SemanticScore:
    judgments: list[SemanticJudgment] = field(default_factory=list)

    def _classified(self) -> list[SemanticJudgment]:
        return [j for j in self.judgments if j.label is not None]

    @property
    def positive_rate(self) -> float:
        classified = self._classified()

        if not classified:
            return 0.0

        return sum(j.label for j in classified) / len(classified)

    @property
    def negative_rate(self) -> float:
        return 1.0 - self.positive_rate

    @property
    def nr(self) -> int:
        return len(self.judgments)

    @property
    def nr_unclassified(self) -> int:
        return sum(1 for j in self.judgments if j.label is None)

    def positive_ids(self) -> list[int]:
        return [i for i, j in enumerate(self.judgments) if j.label == 1]

    def negative_ids(self) -> list[int]:
        return [i for i, j in enumerate(self.judgments) if j.label == 0]

    def unclassified_ids(self) -> list[int]:
        return [i for i, j in enumerate(self.judgments) if j.label is None]

    def __repr__(self) -> str:
        return (
            f"SemanticScore(nr={self.nr}, positive_rate={self.positive_rate:.2%}, "
            f"negative_rate={self.negative_rate:.2%}, nr_unclassified={self.nr_unclassified})"
        )


class SemanticJudge(ABC):
    """Abstract judge that classifies a (command, response) pair against a
    binary semantic property — refusal, truthfulness, code-vulnerability,
    sycophancy, etc.

    The property-and-backend split is intentional. Subclass `APIJudge` or
    `LocalJudge` (in `evaluate.api` / `evaluate.local`) to plug in a backend,
    and parameterize that subclass with property-defining bits (system prompt,
    tool schema) to specialize it. `ClaudeRefusalJudge` is the worked example.

    Only `judge` is abstract; `judge_batch` has a default serial implementation
    that calls `judge` per example.
    """

    @abstractmethod
    def judge(self, command: str, response: str) -> int | None:
        """Return 1 if `response` exhibits the property, 0 if it does not,
        None if unclassifiable."""
        ...

    def judge_batch(
        self,
        commands: list[str],
        responses: list[str],
    ) -> SemanticScore:
        """Judge a batch of command/response pairs serially.

        Args:
            commands: User commands that were sent to the model under test.
            responses: Corresponding model responses.

        Returns:
            SemanticScore containing one SemanticJudgment per pair.
        """
        if len(commands) != len(responses):
            raise ValueError(
                f"commands and responses must have the same length, "
                f"got {len(commands)} and {len(responses)}"
            )

        results = []

        for cmd, resp in zip(commands, responses):
            label = self.judge(cmd, resp)

            results.append(SemanticJudgment(
                command=cmd,
                response=resp,
                label=label,
            ))

        return SemanticScore(judgments=results)
