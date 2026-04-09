from src.utils import clean_text


def test_clean_text():
    text = "hello   world\n\nthis   is test"
    assert clean_text(text) == "hello world this is test"
