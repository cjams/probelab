from __future__ import annotations

import time

from abc import abstractmethod
from typing import Any, Callable

import anthropic

from .base import SemanticJudge

_TRANSIENT_ERRORS = (
    anthropic.APIConnectionError,
    anthropic.APITimeoutError,
    anthropic.InternalServerError,
    anthropic.RateLimitError,
)

# OverloadedError (HTTP 529) is its own class in newer anthropic SDKs but
# falls under InternalServerError in older ones. Add it defensively.
if hasattr(anthropic, "OverloadedError"):
    _TRANSIENT_ERRORS = _TRANSIENT_ERRORS + (anthropic.OverloadedError,)


def _is_retryable(exc: BaseException) -> bool:
    """Return True if `exc` is a transient API condition we should retry on
    rather than give up. Catches by both class hierarchy and HTTP status code,
    so a 529 / 502 / 503 surfaced via a generic APIStatusError is still
    handled even when the SDK has no specific subclass for it."""
    if isinstance(exc, _TRANSIENT_ERRORS):
        return True

    if isinstance(exc, anthropic.APIStatusError):
        status = getattr(exc, "status_code", None)

        if status is not None and (status >= 500 or status == 429):
            return True

    return False


def _call_with_outer_retry(
    fn: Callable[[], Any],
    *,
    label: str,
    max_retries: int = 3,
    base_delay: float = 2.0,
) -> Any:
    """Run `fn()` with exponential-backoff retries on transient errors.

    The Anthropic SDK already retries internally (we set max_retries=5 on
    the client). This is a *second* layer for sustained outages where the
    inner retries get exhausted before the API recovers — common for 529
    Overloaded during long sweeps. Sleeps `base_delay * 2**attempt` between
    attempts (default: 2s, 4s, 8s — total ~14s wasted in the worst case).

    Non-retryable errors (auth, permission, validation) raise immediately
    rather than wasting backoff time.
    """
    last_exc: BaseException | None = None

    for attempt in range(max_retries + 1):
        try:
            return fn()
        except Exception as e:
            if not _is_retryable(e):
                raise

            last_exc = e

            if attempt >= max_retries:
                break

            delay = base_delay * (2 ** attempt)
            print(
                f"[{label}] retryable {type(e).__name__} on outer attempt "
                f"{attempt + 1}/{max_retries + 1}: {e!s}. "
                f"Sleeping {delay:.1f}s.",
                flush=True,
            )
            time.sleep(delay)

    assert last_exc is not None
    raise last_exc


# Hard-coded multi-sample tool used by `ClaudeJudge.judge_many`. Returns one
# binary label per input sample as a list, in the same order the user prompt
# presented them. Works for any binary-classification subclass (refusal,
# truthfulness, sycophancy, ...) without needing to expose a separate
# constructor parameter — the per-sample tool_schema/output_field still
# governs single-sample calls; this one is only used for batched calls.
_BATCH_TOOL_NAME    = "classify_batch"
_BATCH_OUTPUT_FIELD = "labels"
_BATCH_TOOL_SCHEMA: dict = {
    "description": (
        "Classify a batch of (user command, AI response) pairs against the "
        "binary semantic property described in the system prompt. Return one "
        "label per sample in the input order."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            _BATCH_OUTPUT_FIELD: {
                "type": "array",
                "items": {"type": "integer", "enum": [0, 1]},
                "description": (
                    "Per-sample labels. Length MUST match the number of "
                    "samples in the input. Use 1 for the positive class, "
                    "0 for the negative class."
                ),
            },
        },
        "required": [_BATCH_OUTPUT_FIELD],
    },
}


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
        cls_name = type(self).__name__

        def call():
            return self._client.messages.create(
                model=self._model,
                max_tokens=1024,
                system=self.system_prompt,
                tools=[{
                    "name":         self.tool_name,
                    "description":  self.tool_schema.get("description", ""),
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

        # NEVER let an API exception escape — a single bad call should not
        # kill the surrounding pipeline. Convert anything unexpected into
        # an unclassified result and keep going.
        try:
            message = _call_with_outer_retry(call, label=cls_name)
        except Exception as e:
            print(f"[{cls_name}] giving up after retries: "
                  f"{type(e).__name__}: {e!s}. Marking unclassified.",
                  flush=True)
            return None

        if message.stop_reason == "max_tokens":
            print(f"[{cls_name}] response truncated (max_tokens). "
                  f"Marking unclassified.", flush=True)
            return None

        if message.stop_reason == "refusal":
            return None

        for block in message.content:
            if block.type == "tool_use" and block.name == self.tool_name:
                if self.output_field not in block.input:
                    print(f"[{cls_name}] {self.tool_name} returned without "
                          f"{self.output_field!r} (got keys: "
                          f"{list(block.input.keys())}). Marking unclassified.",
                          flush=True)
                    return None

                try:
                    return int(block.input[self.output_field])
                except (TypeError, ValueError):
                    print(f"[{cls_name}] {self.output_field!r} not coercible "
                          f"to int (got {block.input[self.output_field]!r}). "
                          f"Marking unclassified.", flush=True)
                    return None

        print(f"[{cls_name}] no tool_use block in response. Marking unclassified.",
              flush=True)
        return None

    def judge_many(
        self,
        commands: list[str],
        responses: list[str],
    ) -> list[int | None]:
        """Multi-sample Claude call.

        Bundles up to N samples into a single Anthropic API request using
        the hard-coded `classify_batch` tool, which returns a list of N
        binary labels in input order. Failures (transient errors,
        max_tokens, malformed labels, length mismatch) mark the whole
        chunk as unclassified rather than raising — the caller's
        `judge_batch` rebuilds order across chunks regardless.

        For n=1, falls back to the single-sample path so the per-sample
        tool_schema/output_field still applies.
        """
        n = len(commands)

        if n == 0:
            return []

        if n == 1:
            return [self.judge(commands[0], responses[0])]

        # Numbered prompt — making the per-sample boundary explicit makes the
        # length-mismatch failure mode much rarer in practice.
        body = "\n\n".join(
            f"--- Sample {i + 1} ---\n"
            f"User command:\n{cmd}\n\n"
            f"Assistant response:\n{resp}"
            for i, (cmd, resp) in enumerate(zip(commands, responses))
        )
        
        user_content = (
            f"{body}\n\n"
            f"Classify each of the {n} samples above. Return `labels` as a "
            f"list of length {n} in input order."
        )

        cls_name = type(self).__name__

        def call():
            return self._client.messages.create(
                model=self._model,
                # Generous: ~30 tokens per label is overkill but safer than
                # truncating a 50-sample batch on a couple of stray tokens.
                max_tokens=max(512, 32 * n),
                system=self.system_prompt,
                tools=[{
                    "name":         _BATCH_TOOL_NAME,
                    "description":  _BATCH_TOOL_SCHEMA["description"],
                    "input_schema": _BATCH_TOOL_SCHEMA["input_schema"],
                }],
                tool_choice={"type": "tool", "name": _BATCH_TOOL_NAME},
                messages=[{"role": "user", "content": user_content}],
            )

        # NEVER let an API exception escape — judge results being None for
        # a chunk lets the pipeline carry on with whatever signal it has.
        try:
            message = _call_with_outer_retry(call, label=f"{cls_name}.batch")
        except Exception as e:
            print(f"[{cls_name}] batch giving up after retries: "
                  f"{type(e).__name__}: {e!s}. Marking {n} samples unclassified.",
                  flush=True)
            return [None] * n

        if message.stop_reason == "max_tokens":
            print(f"[{cls_name}] batch hit max_tokens (n={n}). "
                  f"Marking all unclassified.")
            return [None] * n

        if message.stop_reason == "refusal":
            return [None] * n

        for block in message.content:
            if block.type != "tool_use" or block.name != _BATCH_TOOL_NAME:
                continue

            if _BATCH_OUTPUT_FIELD not in block.input:
                print(f"[{cls_name}] batch tool returned without "
                      f"{_BATCH_OUTPUT_FIELD!r}. Got keys: "
                      f"{list(block.input.keys())}. Marking all unclassified.")
                return [None] * n

            labels = block.input[_BATCH_OUTPUT_FIELD]

            if not isinstance(labels, list) or len(labels) != n:
                got = len(labels) if isinstance(labels, list) else type(labels).__name__
                print(f"[{cls_name}] batch tool returned {got} labels, "
                      f"expected {n}. Marking all unclassified.")
                return [None] * n

            return [int(l) if l in (0, 1) else None for l in labels]

        print(f"[{cls_name}] batch returned no tool_use block. "
              f"Marking all unclassified.")
        return [None] * n
