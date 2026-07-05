"""Domain-independent reflection helpers for AgentFlow."""


class ReflectionGuide:
    """
    Provides a compact self-check guide for planners.

    This is prompt-level reflection for the baseline ablation: it asks the model
    to verify its next action before producing the JSON decision. It does not
    add domain-specific rules; domain behavior must come from policy and tools.
    """

    def format(self) -> str:
        return """
Before choosing the next JSON action, silently check:
- Is the next response supported by the system/domain policy, conversation, or tool observations?
- Is a tool needed for facts, external state, or domain-specific operations?
- Are required arguments known? If not, ask the user instead of guessing.
- Did any tool return an error or ambiguous result that should change the next action?
- If producing final_answer, is the task actually ready to answer without inventing missing facts?
""".strip()
