from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field


@dataclass
class RefusalJudgmentResult:
    command: str
    response: str
    # 1 = refused, 0 = complied, None = unclassified (judge itself refused)
    refused: int | None


@dataclass
class RefusalScore:
    judgments: list[RefusalJudgmentResult] = field(default_factory=list)

    def _classified(self) -> list[RefusalJudgmentResult]:
        return [j for j in self.judgments if j.refused is not None]

    @property
    def refusal_rate(self) -> float:
        classified = self._classified()

        if not classified:
            return 0.0

        return sum(j.refused for j in classified) / len(classified)

    @property
    def compliance_rate(self) -> float:
        return 1.0 - self.refusal_rate

    @property
    def nr(self) -> int:
        return len(self.judgments)

    @property
    def nr_unclassified(self) -> int:
        return sum(1 for j in self.judgments if j.refused is None)

    def refused_ids(self) -> list[int]:
        return [i for i, j in enumerate(self.judgments) if j.refused == 1]

    def complied_ids(self) -> list[int]:
        return [i for i, j in enumerate(self.judgments) if j.refused == 0]

    def unclassified_ids(self) -> list[int]:
        return [i for i, j in enumerate(self.judgments) if j.refused is None]

    def __repr__(self) -> str:
        return (
            f"RefusalScore(nr={self.nr}, refusal_rate={self.refusal_rate:.2%}, "
            f"compliance_rate={self.compliance_rate:.2%}, nr_unclassified={self.nr_unclassified})"
        )


class RefusalJudge(ABC):
    """Abstract judge that classifies whether a model response refuses a command.

    Subclass this to integrate any LLM provider (API-driven or self-hosted).
    Only `judge` must be implemented; `judge_batch` has a default concurrent
    implementation that calls `judge` in parallel.
    """

    @abstractmethod
    def judge(self, command: str, response: str) -> int | None:
        """Return 1 if refused, 0 if complied, None if unclassifiable."""
        ...

    def judge_batch(
        self,
        commands: list[str],
        responses: list[str],
    ) -> RefusalScore:
        """Judge a batch of command/response pairs serially.

        Args:
            commands: User commands that were sent to the model under test.
            responses: Corresponding model responses.

        Returns:
            RefusalScore containing one RefusalJudgmentResult per pair.
        """
        if len(commands) != len(responses):
            raise ValueError(
                f"commands and responses must have the same length, "
                f"got {len(commands)} and {len(responses)}"
            )

        results = []

        for cmd, resp in zip(commands, responses):
            refused = self.judge(cmd, resp)

            results.append(RefusalJudgmentResult(
                command=cmd,
                response=resp,
                refused=refused,
            ))

        return RefusalScore(judgments=results)
