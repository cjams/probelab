from probelab.dataset.base import Example, ProbeDataset
from typing import Callable

def _apply_instructionify(instructionify: bool, example: Example) -> str:
    if instructionify and example.instructionify is not None:
        return example.instructionify(example.text)
    return example.text

class ChatFormatter:
    """
    Formats a single example or ProbeDataset as a chat-template prompt.

    Optionally applies instructionify to transform the raw example text before
    passing it to the model's chat template via apply_chat_template.

    Args:
        tokenizer:              HuggingFace tokenizer with apply_chat_template.
        instructionify:         Transforms raw example text into prompt-ready
                                form. None means use the text as-is, sourced from
                                dataset.instructionify.
        system_prompt:          Fixed string, or a per-example callable for
                                cases like strategic deception where the system
                                prompt varies across examples.
        add_generation_prompt:  Whether to append the assistant turn opener.
    """

    def __init__(
        self,
        tokenizer,
        instructionify: bool = False,
        system_prompt: str | Callable[[Example], str] | None = None,
        add_generation_prompt: bool = True,
    ):
        self.tokenizer = tokenizer
        self.instructionify = instructionify
        self.system_prompt = system_prompt
        self.add_generation_prompt = add_generation_prompt

    def format(self, data: Example | ProbeDataset) -> str | list[str]:
        if isinstance(data, ProbeDataset):
            return [self._format_one(ex) for ex in data]
        
        return self._format_one(data)

    def _format_one(self, example: Example) -> str:
        messages = []

        if self.system_prompt is not None:
            sp = self.system_prompt(example) if callable(self.system_prompt) else self.system_prompt
            messages.append({"role": "system", "content": sp})

        messages.append({"role": "user", "content": _apply_instructionify(self.instructionify, example)})

        return self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=self.add_generation_prompt,
        )


class FewShotFormatter:
    """
    Formats an example as a completion-style few-shot prompt (no chat template).

    Concatenates formatted shot examples followed by a query stub. Used by
    papers like Geometry of Truth and Mixture of Corrections that probe
    activations from plain completion prompts.

    Args:
        shots:          ProbeDataset whose examples appear as in-context shots.
        instructionify: Transforms raw example text into prompt-ready form.
                        None means use the text as-is.
        pos_completion: Label string for positive examples, e.g. "TRUE".
        neg_completion: Label string for negative examples, e.g. "FALSE".
        shot_template:  Format string for each shot. Placeholders: {text},
                        {label}. E.g. "Input: {text}\\nAnswer: {label}".
        query_template: Format string for the final query (no label). Placeholder:
                        {text}. E.g. "Input: {text}\\nAnswer:".
        separator:      String inserted between each shot and the query.
    """

    def __init__(
        self,
        shots: ProbeDataset,
        pos_completion: str,
        neg_completion: str,
        shot_template: str,
        query_template: str,
        instructionify: bool = False,
        separator: str = "\n\n",
    ):
        self.shots = shots
        self.instructionify = instructionify
        self.pos_completion = pos_completion
        self.neg_completion = neg_completion
        self.shot_template = shot_template
        self.query_template = query_template
        self.separator = separator

    def format(self, data: Example | ProbeDataset) -> str | list[str]:
        if isinstance(data, ProbeDataset):
            return [self._format_one(ex) for ex in data]
        return self._format_one(data)

    def _format_one(self, example: Example) -> str:
        parts = []

        for shot in self.shots:
            label = self.pos_completion if shot.label else self.neg_completion
            parts.append(self.shot_template.format(
                text=_apply_instructionify(self.instructionify, shot),
                label=label,
            ))

        parts.append(self.query_template.format(
            text=_apply_instructionify(self.instructionify, example),
        ))

        return self.separator.join(parts)


class ChatFewShotFormatter:
    """
    Formats an example as a chat-template prompt with in-context shots.

    Two shot placement modes:

    shots_as_turns=True (default):
        Each shot becomes a user/assistant exchange in the chat history.
        The query is appended as the final user message.

            [system]
            [user: shot 1] [assistant: TRUE]
            [user: shot 2] [assistant: FALSE]
            ...
            [user: query]

    shots_as_turns=False (packed):
        All shots are formatted with shot_template and concatenated into a
        single user message, with the query appended via query_template.
        Useful when the model's chat template doesn't handle many-turn
        histories gracefully, or when the shots are part of a larger
        structured user message.

            [system]
            [user: <shot 1>\\n\\n<shot 2>\\n\\n...\\n\\n<query>]

    Args:
        tokenizer:              HuggingFace tokenizer with apply_chat_template.
        shots:                  ProbeDataset whose examples appear as shots.
        instructionify:         Transforms raw example text into prompt-ready
                                form. None means use the text as-is.
        pos_completion:         Completion string for positive shots, e.g. "TRUE".
        neg_completion:         Completion string for negative shots, e.g. "FALSE".
        system_prompt:          Fixed string or per-example callable.
        shots_as_turns:         True = multi-turn chat history, False = packed.
        shot_template:          Used when shots_as_turns=False. Placeholders:
                                {text}, {label}.
        query_template:         Used when shots_as_turns=False. Placeholder:
                                {text}. Appended after the shots.
        separator:              Separator between shots when shots_as_turns=False.
        add_generation_prompt:  Whether to append the assistant turn opener.
    """

    def __init__(
        self,
        tokenizer,
        shots: ProbeDataset,
        pos_completion: str,
        neg_completion: str,
        instructionify: bool = False,
        system_prompt: str | Callable[[Example], str] | None = None,
        shots_as_turns: bool = True,
        shot_template: str | None = None,
        query_template: str | None = None,
        separator: str = "\n\n",
        add_generation_prompt: bool = True,
    ):
        if not shots_as_turns and (shot_template is None or query_template is None):
            raise ValueError(
                "shot_template and query_template are required when shots_as_turns=False"
            )

        self.tokenizer = tokenizer
        self.shots = shots
        self.instructionify = instructionify
        self.pos_completion = pos_completion
        self.neg_completion = neg_completion
        self.system_prompt = system_prompt
        self.shots_as_turns = shots_as_turns
        self.shot_template = shot_template
        self.query_template = query_template
        self.separator = separator
        self.add_generation_prompt = add_generation_prompt

    def format(self, data: Example | ProbeDataset) -> str | list[str]:
        if isinstance(data, ProbeDataset):
            return [self._format_one(ex) for ex in data]
        return self._format_one(data)

    def _format_one(self, example: Example) -> str:
        sp = None
        if self.system_prompt is not None:
            sp = self.system_prompt(example) if callable(self.system_prompt) else self.system_prompt

        if self.shots_as_turns:
            messages = []
            if sp is not None:
                messages.append({"role": "system", "content": sp})

            for shot in self.shots:
                completion = self.pos_completion if shot.label else self.neg_completion
                messages.append({"role": "user", "content": _apply_instructionify(self.instructionify, shot)})
                messages.append({"role": "assistant", "content": completion})

            messages.append({"role": "user", "content": _apply_instructionify(self.instructionify, example)})
        else:
            parts = []
            for shot in self.shots:
                label = self.pos_completion if shot.label else self.neg_completion
                parts.append(self.shot_template.format(
                    text=_apply_instructionify(self.instructionify, shot),
                    label=label,
                ))

            parts.append(self.query_template.format(
                text=_apply_instructionify(self.instructionify, example),
            ))

            messages = []
            if sp is not None:
                messages.append({"role": "system", "content": sp})

            messages.append({"role": "user", "content": self.separator.join(parts)})

        return self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=self.add_generation_prompt,
        )
