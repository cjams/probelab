from __future__ import annotations

from abc import abstractmethod

from .base import SemanticJudge


class LocalJudge(SemanticJudge):
    """Abstract base for self-hosted semantic judges — anything outside the
    third-party API providers covered by `APIJudge`.

    Subclasses bake in the property (system prompt, label set, or whatever
    the inference path needs) and the inference path itself. `judge` returns
    1 / 0 / None per `SemanticJudge.judge`.
    """

    def __init__(self, system_prompt: str | None = None) -> None:
        self.system_prompt = system_prompt

    @abstractmethod
    def judge(self, command: str, response: str) -> int | None:
        ...
