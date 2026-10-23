from __future__ import annotations

import json
import re

from contextwaise import shape
from contextwaise._shapers import MAX_PART_TOKENS, sections, tokens
from tests.stubs import TEST_RUN


def test_passing_tests_go_and_failures_stay() -> None:
    shaped = shape(TEST_RUN)
    assert shaped.kind == "test"
    assert "module 5 works" not in shaped.text
    assert "expected 90 to equal 81" in shaped.text


def test_repeated_log_lines_collapse_onto_one() -> None:
    log = "\n".join(f"2026-10-01 10:00:{i:02d} INFO heartbeat ok" for i in range(50))
    assert shape(log).text.endswith("heartbeat ok  (×50)")


def test_error_lines_are_never_collapsed() -> None:
    log = "\n".join(f"2026-10-01 10:00:{i:02d} ERROR disk full" for i in range(20))
    assert shape(log).text.count("disk full") == 20


def test_empty_json_fields_go() -> None:
    assert shape('{"id":1,"note":null,"tags":[]}').text == '{"id":1}'


def test_html_becomes_text_without_navigation_scripts_banners_or_asides() -> None:
    page = (
        "<html><head><title>Filtering at scale</title></head><body>"
        '<nav>Menu</nav><div class="cookie-banner">We use cookies. Accept all</div>'
        "<article><p>Qdrant filters during search.</p></article>"
        "<aside>Subscribe to our newsletter</aside><script>x()</script></body></html>"
    )
    assert shape(page).text == "Filtering at scale\nQdrant filters during search."


def test_listings_group_by_folder_and_collapse_numbered_series() -> None:
    paths = [
        *(f"src/admin/components/Widget{i}.tsx" for i in range(30)),
        "src/admin/index.ts",
        "src/auth/middleware.ts",
        "src/auth/session.ts",
        "README.md",
        "a/b/x1.ts",
        "a/b/x5.ts",
        *(f"src/f{i}/index.ts" for i in range(20)),
    ]
    shaped = shape("\n".join(paths))
    assert shaped.kind == "listing"
    assert re.search(r"^src/admin/components/  Widget\{0\.\.29\}\.tsx$", shaped.text, re.M)
    assert re.search(r"^src/auth/  middleware\.ts, session\.ts$", shaped.text, re.M)
    assert re.search(r"^a/b/  x1\.ts, x5\.ts$", shaped.text, re.M), "non-contiguous numbers stay listed"
    assert re.search(r"^\./  README\.md$", shaped.text, re.M)


def test_listings_keep_zero_padding() -> None:
    paths = [
        *(f"img/frame{i:03d}.png" for i in range(1, 6)),
        "img/frame010.png",
        *(f"x/y{i}/a.ts" for i in range(20)),
    ]
    text = shape("\n".join(paths)).text
    assert "frame00{1..5}.png" in text
    assert "frame010.png" in text


def test_windows_line_endings_keep_their_content() -> None:
    shaped = shape("first line\r\nsecond line\r\n")
    assert shaped.text == "first line\nsecond line"
    assert shaped.notes == ()


def test_terminal_noise_goes() -> None:
    shaped = shape("\x1b[32mok\x1b[0m\n[========    ] 60%\nloading 10%\rloading 100%")
    assert shaped.text == "ok\nloading 100%"
    assert "removed 1 progress-bar lines" in shaped.notes
    assert "removed terminal color codes" in shaped.notes


def test_only_real_whitespace_earns_the_compaction_note() -> None:
    assert "compacted JSON whitespace" not in shape('{"id":1,"tags":["a"]}').notes
    assert "compacted JSON whitespace" in shape('{"id": 1, "tags": ["a"]}').notes


def test_tables_and_dumps_are_line_oriented_but_prose_and_code_are_not() -> None:
    csv = "\n".join(["date,users,calls", *(f"2026-09-{d:02d},{d * 3},{d * 100}" for d in range(1, 29))])
    assert shape(csv).kind == "lines"
    assert shape(csv).text == csv, "rows are kept as they are: no collapsing of near-identical lines"
    code = "\n\n".join(f"def f{i}(x):\n    y = x + {i}\n    return y * 2" for i in range(12))
    assert shape(code).kind == "text"


def test_a_big_json_value_is_cut_into_path_named_parts() -> None:
    schema = {"input_schema": {"properties": {f"field_{i}": {"description": "d" * 300} for i in range(40)}}}
    shaped = shape(json.dumps(schema))
    units = sections(shaped)
    assert len(units) == 40
    assert units[3].startswith('{"input_schema.properties.field_3":')


def test_an_mcp_result_is_unwrapped_and_the_json_string_inside_cut_like_json() -> None:
    inner = json.dumps({"rows": [{"id": i, "text": "t" * 400} for i in range(30)]})
    shaped = shape(json.dumps({"content": [{"type": "text", "text": inner}]}))
    assert shaped.kind == "json"
    assert "unwrapped the MCP result" in shaped.notes
    assert len(sections(shaped)) == 30
    error = json.dumps({"content": [{"type": "text", "text": "boom"}], "isError": True})
    assert shape(error).text == "Error: boom"


def test_no_part_is_bigger_than_the_limit_even_on_one_long_line() -> None:
    xml = (
        "<orders>"
        + "".join(f'<order id="ORD-{i}"><c>Customer {i}</c></order>' for i in range(500))
        + "</orders>"
    )
    units = sections(shape(xml))
    assert len(units) > 5
    assert all(tokens(u) <= MAX_PART_TOKENS for u in units)


def test_fenced_or_introduced_json_is_cleaned_as_json() -> None:
    data = json.dumps([{"id": 1, "note": None}])
    fenced = shape(f"```json\n{data}\n```")
    assert (fenced.kind, fenced.text) == ("json", '[{"id":1}]')
    introduced = shape(f"Found 1 order:\n{data}")
    assert introduced.kind == "json"
    assert 'text before the JSON: "Found 1 order:"' in introduced.notes
    assert shape(f"Error: quota exceeded\n{data}").kind != "json", "an error before the JSON stays visible"
