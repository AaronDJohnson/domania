"""README.md's python blocks run, in order, in one namespace."""

from __future__ import annotations

import re
from pathlib import Path

import pytest

README = Path(__file__).resolve().parents[1] / "README.md"
#: A fenced ```python block: the opening fence, the body, the closing fence, each on its own line.
FENCE = re.compile(r"^```python[ \t]*\n(.*?)^```[ \t]*$", re.MULTILINE | re.DOTALL)


def run_python_blocks(text: str, name: str, namespace: dict | None = None) -> int:
    """Run every ```python block of the Markdown ``text`` in order, in one ``namespace``.

    Returns the number of blocks run.  A block that raises is reported as an
    ``AssertionError`` naming the block (1-based) and the line of its opening fence in
    ``name``, chained to the original exception.
    """
    ns = {"__name__": "__readme__"} if namespace is None else namespace
    blocks = list(FENCE.finditer(text))
    for i, block in enumerate(blocks, start=1):
        line = text.count("\n", 0, block.start()) + 1      # the opening fence
        where = f"{name} python block {i} of {len(blocks)} (line {line})"
        code = compile(block.group(1), f"<{where}>", "exec")
        try:
            exec(code, ns)
        except Exception as exc:
            raise AssertionError(f"{where} raised {type(exc).__name__}: {exc}") from exc
    return len(blocks)


def test_readme_python_blocks_run():
    text = README.read_text()
    n = run_python_blocks(text, "README.md")
    assert n >= 1 and n == text.count("```python")          # the fence pattern missed none


def test_a_failing_block_is_named_by_its_index():
    """Block 2 reads block 1's name (one namespace) and raises; block 3 never runs, and the
    ```bash block in between is not python."""
    text = ("# Title\n\n```python\na = 1\n```\n\nSome prose.\n\n```bash\nexit 1\n```\n\n"
            "```python\nb = 1 / (a - 1)\n```\n\n```python\nc = 3\n```\n")
    ns: dict = {}
    with pytest.raises(AssertionError, match=r"^SYNTHETIC\.md python block 2 of 3 \(line 13\) "
                                             r"raised ZeroDivisionError") as info:
        run_python_blocks(text, "SYNTHETIC.md", ns)
    assert isinstance(info.value.__cause__, ZeroDivisionError)
    assert ns["a"] == 1 and "b" not in ns and "c" not in ns
