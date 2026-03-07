import pytest

from evalkit.prompts import PromptTemplate
from evalkit.types import Task


@pytest.fixture
def qa_prompt() -> PromptTemplate:
    return PromptTemplate(name="qa", system="Answer briefly.", user="Q: {question}")


@pytest.fixture
def arithmetic_tasks() -> list[Task]:
    return [
        Task(id="a1", category="math", input={"question": "2+2"}, expected=4, grader={"type": "numeric"}),
        Task(id="a2", category="math", input={"question": "3*3"}, expected=9, grader={"type": "numeric"}),
        Task(id="s1", category="text", input={"question": "capital of France"}, expected="Paris",
             grader={"type": "normalized"}),
    ]
