from __future__ import annotations

from abc import ABC, abstractmethod
from concurrent.futures import ThreadPoolExecutor
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

    def judge_many(
        self,
        commands: list[str],
        responses: list[str],
    ) -> list[int | None]:
        """Judge multiple (command, response) pairs in a single API call.

        The default implementation falls back to per-sample `judge()`, so
        existing subclasses keep working without modification. Subclasses
        whose backend supports multi-sample requests (e.g. an LLM tool-use
        call returning a list of labels) should override this to amortize
        per-call overhead — system prompt, tool schema, network round trip —
        across the chunk.

        Returns one label per input sample, in the same order as the input.
        On any failure (transient error, malformed response, length mismatch)
        the implementation should return `[None] * len(commands)` rather
        than raising, so a single bad chunk doesn't sink the whole batch.
        """
        return [self.judge(cmd, resp) for cmd, resp in zip(commands, responses)]

    def judge_batch(
        self,
        commands: list[str],
        responses: list[str],
        *,
        samples_per_call: int = 10,
        max_concurrency: int = 8,
    ) -> SemanticScore:
        """Judge a batch of (command, response) pairs with two dimensions of
        throughput:

        - `samples_per_call` (k): pack k samples into each `judge_many()` call
          so the system prompt and tool schema overhead is amortized.
        - `max_concurrency` (M): dispatch chunks across up to M worker threads.

        Total throughput ≈ M × k samples per round-trip-time. Tune both for
        your provider's rate limits — most API SDKs are thread-safe and use
        connection pooling, so M=8 with k=10 is a reasonable default.

        Set `samples_per_call=1` for one-sample-per-call (useful when the
        subclass has no batched-API path), or `max_concurrency=1` for serial
        execution (useful for debugging).
        """
        if len(commands) != len(responses):
            raise ValueError(
                f"commands and responses must have the same length, "
                f"got {len(commands)} and {len(responses)}"
            )

        n = len(commands)

        if n == 0:
            return SemanticScore(judgments=[])

        # Chunk into (start_idx, cmds, resps) so we can write results back in
        # order regardless of completion order under concurrency.
        chunks: list[tuple[int, list[str], list[str]]] = [
            (i, commands[i:i + samples_per_call], responses[i:i + samples_per_call])
            for i in range(0, n, samples_per_call)
        ]

        labels: list[int | None] = [None] * n

        def run(chunk: tuple[int, list[str], list[str]]) -> tuple[int, list[int | None]]:
            start, cmds, resps = chunk
            return start, self.judge_many(cmds, resps)

        if max_concurrency <= 1 or len(chunks) <= 1:
            for chunk in chunks:
                start, chunk_labels = run(chunk)

                for offset, label in enumerate(chunk_labels):
                    labels[start + offset] = label
        else:
            with ThreadPoolExecutor(max_workers=max_concurrency) as ex:
                for start, chunk_labels in ex.map(run, chunks):
                    for offset, label in enumerate(chunk_labels):
                        labels[start + offset] = label

        return SemanticScore(judgments=[
            SemanticJudgment(command=cmd, response=resp, label=label)
            for cmd, resp, label in zip(commands, responses, labels)
        ])
