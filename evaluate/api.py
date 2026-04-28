from __future__ import annotations

from abc import abstractmethod

import anthropic

from .base import SemanticJudge

_TRANSIENT_ERRORS = (
    anthropic.APIConnectionError,
    anthropic.APITimeoutError,
    anthropic.InternalServerError,
    anthropic.RateLimitError,
)


class APIJudge(SemanticJudge):
    """Abstract base for API-backed semantic judges.

    Holds the property-defining bits — the system prompt that tells the judge
    what to look for, and a tool/function-call schema that forces a binary
    output. Subclasses pick a specific provider (Anthropic, OpenAI, etc.)
    and implement `judge` against it using these fields.

    The two-axis split (provider × property) lets you reuse one provider
    class across many properties (refusal, truthfulness, sycophancy, ...)
    by just swapping the system prompt and tool schema. See
    `ClaudeRefusalJudge` for the worked example.
    """

    def __init__(
        self,
        system_prompt: str,
        tool_name: str,
        tool_schema: dict,
        output_field: str,
    ) -> None:
        self.system_prompt = system_prompt
        self.tool_name = tool_name
        self.tool_schema = tool_schema
        self.output_field = output_field

    @abstractmethod
    def judge(self, command: str, response: str) -> int | None:
        """Subclasses implement the provider-specific call. Return 1, 0, or
        None per `SemanticJudge.judge`."""
        ...


class ClaudeJudge(APIJudge):
    """Claude-backed semantic judge. Concrete over the Anthropic API,
    abstract over the property — instantiate directly with a custom
    (system_prompt, tool_*) for a one-off, or subclass with constants
    baked in for a named property (see `ClaudeRefusalJudge`).
    """

    def __init__(
        self,
        system_prompt: str,
        tool_name: str,
        tool_schema: dict,
        output_field: str,
        model: str = "claude-sonnet-4-6",
        api_key: str | None = None,
        max_retries: int = 5,
    ) -> None:
        super().__init__(
            system_prompt=system_prompt,
            tool_name=tool_name,
            tool_schema=tool_schema,
            output_field=output_field,
        )
        self._model = model
        self._max_retries = max_retries

        # The SDK's built-in retries handle 5xx/429 already; bumping max_retries
        # here extends that without adding a second retry loop.
        client_kwargs: dict = {"max_retries": max_retries}

        if api_key:
            client_kwargs["api_key"] = api_key

        self._client = anthropic.Anthropic(**client_kwargs)

    def judge(self, command: str, response: str) -> int | None:
        try:
            message = self._client.messages.create(
                model=self._model,
                max_tokens=1024,
                system=self.system_prompt,
                tools=[{
                    "name": self.tool_name,
                    "description": self.tool_schema.get("description", ""),
                    "input_schema": self.tool_schema["input_schema"],
                }],
                tool_choice={"type": "tool", "name": self.tool_name},
                messages=[
                    {
                        "role": "user",
                        "content": (
                            f"User command:\n{command}\n\n"
                            f"Assistant response:\n{response}"
                        ),
                    }
                ],
            )
        except _TRANSIENT_ERRORS as e:
            print(f"[{type(self).__name__}] transient API error after retries: {e!r}. Marking unclassified.")
            return None

        if message.stop_reason == "max_tokens":
            raise RuntimeError(
                f"Judge response truncated (max_tokens reached). "
                f"Content blocks: {message.content}"
            )

        if message.stop_reason == "refusal":
            return None

        for block in message.content:
            if block.type == "tool_use" and block.name == self.tool_name:
                if self.output_field not in block.input:
                    raise RuntimeError(
                        f"{self.tool_name} tool called with unexpected input keys: {list(block.input.keys())}. "
                        f"Full input: {block.input}. Stop reason: {message.stop_reason}. "
                        f"All content blocks: {message.content}"
                    )

                return int(block.input[self.output_field])

        raise RuntimeError(
            f"Judge returned no tool_use block. Full response: {message}"
        )
