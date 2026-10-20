from assay.metrics import Heading, parse_markdown

DOCUMENT = """## `wrap_text`

Wraps `text` to `width` columns. Use ``a`b`` for odd names.

### Example

- Call it like this:

  ```python
  # `not_a_span` inside a block
  wrap_text("hi", width=10)
  ```

~~~
## not a heading
~~~
"""


def test_separates_inline_code_from_code_blocks() -> None:
    parts = parse_markdown(DOCUMENT)

    assert parts.inline_code == ("wrap_text", "text", "width", "a`b")
    assert len(parts.code_blocks) == 2
    assert 'wrap_text("hi", width=10)' in parts.code_blocks[0]


def test_headings_inside_code_blocks_are_ignored() -> None:
    assert parse_markdown(DOCUMENT).headings == (
        Heading(level=2, text="`wrap_text`"),
        Heading(level=3, text="Example"),
    )


def test_prose_has_no_code_or_headings() -> None:
    prose = parse_markdown(DOCUMENT).prose

    assert "Wraps" in prose
    assert "Call it like this" in prose
    assert "width" not in prose
    assert "Example" not in prose
    assert "not_a_span" not in prose


def test_closing_hashes_and_unclosed_fences() -> None:
    parts = parse_markdown("## Title ##\n```python\nopen = True\n")

    assert parts.headings == (Heading(level=2, text="Title"),)
    assert parts.code_blocks == ()
