"""The exact prompt text sent to every model (identical for edge and cloud)."""

from __future__ import annotations

from eclm.types import Question

LETTERS = "ABCD"

GSM8K_TEMPLATE = (
    "Solve the problem. Show brief reasoning, then end with a final line "
    "'Answer: <number>'.\n\n{question}"
)
MMLU_TEMPLATE = (
    "Answer the multiple-choice question. Think briefly, then end with a final line "
    "'Answer: <letter>'.\n\n{question}\n{options}"
)
MOCK_GOLD_MARKER = "[[gold:{gold}]]"


def build_prompt(q: Question, mock: bool = False) -> str:
    """Build the prompt for a question.

    With mock=True a hidden gold marker is appended; only MockClient reads it.
    Real prompts never contain it.
    """
    if q.dataset == "gsm8k":
        text = GSM8K_TEMPLATE.format(question=q.question)
    else:
        options = "\n".join(f"{LETTERS[i]}. {c}" for i, c in enumerate(q.choices))
        text = MMLU_TEMPLATE.format(question=q.question, options=options)
    if mock:
        text += "\n" + MOCK_GOLD_MARKER.format(gold=q.gold)
    return text
