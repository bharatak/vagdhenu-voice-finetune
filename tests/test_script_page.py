from mv.script_page import render_html
from mv.text import load_sarga


def test_page_has_lines_and_numbers():
    html = render_html(1)
    first = load_sarga(1)[0]
    assert first.hemistichs[0].deva in html
    assert "Sarga 1" in html
    assert 'data-verse="1"' in html


def test_suspect_verse_shown_as_skip():
    html = render_html(1)
    assert 'class="verse skip" data-verse="92"' in html


def test_start_verse():
    html = render_html(1, start=40)
    assert 'data-verse="39"' not in html and 'data-verse="40"' in html


def test_telugu_script_shows_same_lines_in_telugu():
    import re
    page = render_html(1, script="telugu")
    lines = re.findall(r'<p class="line">(.*?)</p>', page)
    assert lines and all(re.search(r"[ఀ-౿]", l) for l in lines)
    assert not any(re.search(r"[ऀ-ॿ]", l.replace("।", "").replace("॥", "")) for l in lines)
