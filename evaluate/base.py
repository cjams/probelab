from __future__ import annotations

from abc import ABC, abstractmethod
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field


@dataclass
class RefusalJudgmentResult:
    command: str
    response: str
    # 1 = refused, 0 = complied
    refused: int


@dataclass
class RefusalScore:
    judgments: list[RefusalJudgmentResult] = field(default_factory=list)

    @property
    def refusal_rate(self) -> float:
        if not self.judgments:
            return 0.0

        return sum(j.refused for j in self.judgments) / len(self.judgments)

    @property
    def compliance_rate(self) -> float:
        return 1.0 - self.refusal_rate

    @property
    def n(self) -> int:
        return len(self.judgments)

    def refused_ids(self) -> list[int]:
        return [i for i, j in enumerate(self.judgments) if j.refused]

    def complied_ids(self) -> list[int]:
        return [i for i, j in enumerate(self.judgments) if not j.refused]

    def __repr__(self) -> str:
        return (
            f"RefusalScore(n={self.n}, refusal_rate={self.refusal_rate:.2%}, "
            f"compliance_rate={self.compliance_rate:.2%})"
        )


class RefusalJudge(ABC):
    """Abstract judge that classifies whether a model response refuses a command.

    Subclass this to integrate any LLM provider (API-driven or self-hosted).
    Only `judge` must be implemented; `judge_batch` has a default concurrent
    implementation that calls `judge` in parallel.
    """

    @abstractmethod
    def judge(self, command: str, response: str) -> int:
        """Return 1 if the response refuses the command, 0 if it complies."""
        ...

    def judge_batch(
        self,
        commands: list[str],
        responses: list[str],
        max_workers: int = 8,
    ) -> RefusalScore:
        """Judge a batch of command/response pairs, running calls concurrently.

        Args:
            commands: User commands that were sent to the model under test.
            responses: Corresponding model responses.
            max_workers: Maximum number of concurrent judge calls.

        Returns:
            RefusalScore containing one RefusalJudgmentResult per pair.
        """
        if len(commands) != len(responses):
            raise ValueError(
                f"commands and responses must have the same length, "
                f"got {len(commands)} and {len(responses)}"
            )

        results: list[RefusalJudgmentResult] = [None] * len(commands)  # type: ignore[list-item]

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {
                executor.submit(self.judge, cmd, resp): i
                for i, (cmd, resp) in enumerate(zip(commands, responses))
            }

            for future in as_completed(futures):
                i = futures[future]
                refused = future.result()

                results[i] = RefusalJudgmentResult(
                    command=commands[i],
                    response=responses[i],
                    refused=refused,
                )

        return RefusalScore(judgments=results)
