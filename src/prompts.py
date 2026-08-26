PLAIN_SYSTEM_PROMPT = """\
You are an expert therapist solving a multiple-choice question.

Answer the question below by writing only the option letter \
(e.g. "A", "B", "C", "D", or "E"). Output ONLY:
<final>[A-E]</final>
"""

SUMMARIZE = """Summarize your answer to the question below by writing only the option letter \
(e.g. "A", "B", "C", "D", or "E")."""

FEW_SHOT_EXAMPLES = [
    {"role": "user", "content": "Is this a question?\n- (A) Yes\n - (B) No"},
    {"role": "assistant", "content": "A"},
]

# Extra few-shot examples for Opus 4.5 plain mode to reinforce single-letter answers
OPUS_PLAIN_FEW_SHOT_EXAMPLES = [
    {"role": "user", "content": "Is this a question?\n- (A) Yes\n- (B) No"},
    {"role": "assistant", "content": "A"},
    {"role": "user", "content": "Is the sky green?\n- (A) Yes\n- (B) No\n- (C) Sometimes\n- (D) Do not know"},
    {"role": "assistant", "content": "B"},
    {"role": "user", "content": "How likely is rain today?\n- (A) Not at all likely\n- (B) Not very likely\n- (C) Somewhat likely\n- (D) Very likely\n- (E) Do not know"},
    {"role": "assistant", "content": "C"},
]

FEW_SHOT_EXAMPLES_COT = [
    {"role": "user", "content": "Is this a question?\n- (A) Yes\n - (B) No"},
    {"role": "assistant", "content": "<scratchpad>The input is a question, so the answer is A. </scratchpad>\n <final>A</final><rationale>Because it is a question.</rationale>"},
]

# Chain-of-Thought functionality

COT_SYSTEM_PROMPT = """\
You are an expert therapist solving a multiple-choice question. When responding, please:

1. Think step by step in a private scratchpad, evaluating all the previous conversation and considering all the options
2. then, answer the question below by writing only the option letter in the <final></final> field\
(e.g. "A", "B", "C", "D", or "E"). Output ONLY:
<final>[A-E]</final>
<rationale>Concise reasoning (≤120 words)</rationale>

"""

NTCOT_SYSTEM_PROMPT = """\
You are answering a multiple-choice survey item as a generic respondent.
Do not roleplay a therapist or any specific demographic/persona.
Treat the questions as hypotheticals about your willingness as a person.

When responding, please:

1. Think step by step in a private scratchpad, evaluating all the previous conversation and considering all the options
2. then, answer the question below by writing only the option letter in the <final></final> field\
(e.g. "A", "B", "C", "D", or "E"). Output ONLY:
<final>[A-E]</final>
<rationale>Concise reasoning (≤120 words)</rationale>

"""

CLAUDE_REINFORCEMENT_PROMPT = """
\n\n
IMPORTANT: Respond with ONLY the single letter.
Output format:
<final>[A-E]</final>
"""


