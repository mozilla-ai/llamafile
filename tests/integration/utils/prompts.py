"""Shared prompts and answer-checkers for liveness-style integration tests.

These prompts don't test model capability (math, instruction-following) --
they exist to confirm a mode is up and producing a coherent, on-topic
response. Keeping them here means updating a prompt or its accepted answers
happens in one place instead of drifting across test files.
"""

import re

NUMBER_WORDS = {
    0: "zero", 1: "one", 2: "two", 3: "three", 4: "four",
    5: "five", 6: "six", 7: "seven", 8: "eight", 9: "nine", 10: "ten",
}


def contains_number(text: str, n: int) -> bool:
    """Check whether *text* contains *n* as a digit or spelled-out word.

    Matches on word boundaries so checking for 6 doesn't match "sixteen" and
    checking for 2 doesn't match "24".
    """
    forms = [str(n)]
    if n in NUMBER_WORDS:
        forms.append(NUMBER_WORDS[n])
    pattern = r"\b(?:" + "|".join(re.escape(f) for f in forms) + r")\b"
    return re.search(pattern, text, re.IGNORECASE) is not None


class MathPrompt:
    """A simple addition fact used as a liveness/coherence check.

    Not a math-capability test: models may answer in digits or words, so
    the check accepts either.
    """

    def __init__(self, prompt: str, answer: int):
        self.prompt = prompt
        self.answer = answer

    def check(self, text: str) -> bool:
        return contains_number(text, self.answer)

    def describe(self) -> str:
        return f"{self.answer} (digit or word form)"


# Canonical single-shot liveness prompt -- use this one everywhere a test
# just needs to confirm the model produced a sane, on-topic answer.
ADD_2_2 = MathPrompt("What is 2+2?", 4)

# Reserved for test_combined's multi-step test, which makes three sequential
# calls (server, TUI, server) in one test run. Using distinct facts there is
# deliberate: it's how that test catches cross-talk (e.g. the TUI reading the
# server's response, or a stale/cached answer) that reusing ADD_2_2 three
# times would mask, since a "4" would look correct regardless of which call
# it actually came from.
ADD_1_1 = MathPrompt("What is 1+1?", 2)
ADD_3_3 = MathPrompt("What is 3+3?", 6)

# Canonical greeting liveness prompt -- confirms a mode responds with some
# non-empty, coherent output. The exact wording doesn't matter (these tests
# only check the response is non-empty), so one wording is reused everywhere.
GREETING_PROMPT = "Say hello in one word."
