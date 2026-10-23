"""The console's two unenforceable-looking rules, enforced.

`CLAUDE.md` states both as rules and neither had anything checking it, which is
the condition under which a rule becomes a comment. They are asserted against
the files rather than against a running browser because both are properties of
the source: one is "a dangerous API is never called" and the other is "two
documents agree", and a browser could not tell you about either more reliably
than `grep` can.

There is no JavaScript test runner in this project and this does not add one.
These read the files.
"""

from __future__ import annotations

import re

import pytest

from app.core.config import PROJECT_ROOT

JS_DIR = PROJECT_ROOT / "static" / "js"
TEMPLATE = PROJECT_ROOT / "templates" / "index.html"
README = PROJECT_ROOT / "README.md"

# Every way a string can become markup, matched as a *use* rather than as a
# mention. `innerHTML` and `outerHTML` are the obvious ones;
# `insertAdjacentHTML` and `document.write` are the two that get reached for
# when somebody has been told not to use `innerHTML`.
#
# Matching the assignment or the call, not the bare word, for a reason worth
# stating: the docstrings in `dom.js` and the template's header comment both
# name these repeatedly, and a check that cannot tell an explanation from a
# call is a check whose first effect is that somebody deletes the explanation.
# Trying to strip comments instead was the first attempt and it was wrong on
# continuation lines inside a block comment.
UNSAFE_SINKS = {
    "innerHTML": re.compile(r"\.innerHTML\s*(=[^=]|\+=)"),
    "outerHTML": re.compile(r"\.outerHTML\s*(=[^=]|\+=)"),
    "insertAdjacentHTML": re.compile(r"\.insertAdjacentHTML\s*\("),
    "document.write": re.compile(r"\bdocument\s*\.\s*write(ln)?\s*\("),
}


def _js_files() -> list:
    return sorted(JS_DIR.glob("*.js"))


def test_there_are_javascript_modules_to_check():
    """Guards the two tests below against passing on an empty glob."""
    assert len(_js_files()) >= 5


@pytest.mark.parametrize("sink", sorted(UNSAFE_SINKS))
def test_the_console_never_assigns_untrusted_text_as_markup(sink):
    """The rule that makes escaping unnecessary rather than easy to forget.

    Answer text comes from a language model, citation titles from uploaded
    documents, audit rows and scratchpad notes from the database. None of it is
    content this application controls, and the admin reading the audit log is
    exactly the session worth stealing. `static/js/dom.js` builds elements and
    sets `textContent`, so there is no escaping step to omit -- the unsafe
    operation is simply not used.

    `el()` throws on an `html` option rather than ignoring it, which is the
    other half: the convenient wrong thing fails loudly in development.
    """
    pattern = UNSAFE_SINKS[sink]
    offenders = []
    for path in [*_js_files(), TEMPLATE]:
        for number, line in enumerate(
            path.read_text(encoding="utf-8").splitlines(), start=1
        ):
            if pattern.search(line):
                offenders.append(f"{path.name}:{number}: {line.strip()}")

    assert not offenders, f"{sink} is used:\n" + "\n".join(offenders)


# Figures the console's Pipeline page states as fact. It is the one screen
# written for a reader who cannot check it against the code, which makes a
# stale number there worse than a stale number anywhere else: nobody reading it
# is in a position to notice.
#
# Listed as exact strings rather than scraped, so adding a claim to the page
# means adding it here -- which is the moment to ask whether it was measured.
CLAIMED_FIGURES = (
    "11",  # documents
    "31",  # chunks
    "0.15",  # the similarity gate, raw cosine
    "21/21",  # the labelled routing and grading set
    "652",  # offline tests
    "0.099",  # out-of-domain peak, first measurement
    "0.171",  # weakest correct chunk
    "0.521",  # strongest wrong-source chunk
    "0.078",  # out-of-domain peak after the corpus grew
)


@pytest.mark.parametrize("figure", CLAIMED_FIGURES)
def test_every_figure_the_console_claims_also_appears_in_the_readme(figure):
    """The console and the README must state the same numbers.

    They are two descriptions of one measurement and they drift in one
    direction: somebody re-measures, updates the README because that is where
    the table is, and leaves the page that a visitor actually reads saying the
    old figure.

    This is a weak check deliberately -- it asserts the figure appears in both
    files, not that it means the same thing in both. A strong version would
    need the pages to share a data source, and a page of static prose is the
    last thing that should acquire a render path.
    """
    page = TEMPLATE.read_text(encoding="utf-8")
    readme = README.read_text(encoding="utf-8")

    assert figure in page, f"{figure} is no longer claimed on the Pipeline page"
    assert figure in readme, (
        f"the console claims {figure} and the README does not; they are two "
        f"descriptions of one measurement"
    )


def test_the_page_offers_every_view_the_bootstrap_knows_about():
    """A rail button with no section, or a section with no way to reach it.

    Both are silent: the button switches to a view that does not exist and the
    page goes blank, or the section is rendered and unreachable. The two lists
    live in different files -- `TITLES` in `app.js` and the markup here -- so
    nothing else notices.
    """
    page = TEMPLATE.read_text(encoding="utf-8")
    boot = (JS_DIR / "app.js").read_text(encoding="utf-8")

    buttons = set(re.findall(r'data-view="([a-z]+)"', page))
    sections = set(re.findall(r'class="view[^"]*" id="view-([a-z]+)"', page))
    titled = set(re.findall(r'^\s+([a-z]+): "', boot, re.MULTILINE))

    assert buttons == sections, (
        f"rail buttons and view sections disagree: "
        f"{buttons ^ sections}"
    )
    assert buttons <= titled, (
        f"a view has no title in app.js: {sorted(buttons - titled)}"
    )
