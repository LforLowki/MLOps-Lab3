import pytest
from lab1 import preprocessing as pp


@pytest.fixture
def sample_numbers():
    return [1, 2, 3, 4, 5]


def test_remove_missing():
    assert pp.remove_missing([1, None, 2, ""]) == [1, 2]


@pytest.mark.parametrize(
    "values,fill_value,expected",
    [([1, None, 2], 0, [1, 0, 2]), ([None, None], 5, [5, 5])],
)
def test_fill_missing(values, fill_value, expected):
    assert pp.fill_missing(values, fill_value) == expected


def test_remove_duplicates():
    assert pp.remove_duplicates([1, 1, 2, 3, 3]) == [1, 2, 3]


@pytest.mark.parametrize(
    "values,new_min,new_max,expected_len",
    [([1, 2, 3], 0, 1, 3), ([10, 10, 10], 0, 1, 3)],
)
def test_normalize(values, new_min, new_max, expected_len):
    assert len(pp.normalize(values, new_min, new_max)) == expected_len


def test_standardize(sample_numbers):
    result = pp.standardize(sample_numbers)
    assert pytest.approx(sum(result), abs=1e-9) == 0


def test_clip():
    assert pp.clip([1, 5, 10], 2, 6) == [2, 5, 6]


def test_convert_to_int():
    assert pp.convert_to_int(["1", "x", "3"]) == [1, 3]


def test_log_transform():
    assert pp.log_transform([1, 10, -5]) == [0.0, pytest.approx(2.3025, rel=1e-3)]


def test_tokenize_text():
    assert pp.tokenize_text("Hello World!") == ["hello", "world"]


def test_remove_non_alnum_spaces():
    assert pp.remove_non_alnum_spaces("Hi! #123") == "Hi 123"


def test_remove_stopwords():
    text = "this is a test"
    stop = ["this", "is"]
    assert pp.remove_stopwords(text, stop) == "a test"


def test_flatten_list():
    assert pp.flatten_list([[1, 2], [3]]) == [1, 2, 3]


def test_shuffle_list(sample_numbers):
    a = pp.shuffle_list(sample_numbers, seed=42)
    b = pp.shuffle_list(sample_numbers, seed=42)
    assert a == b
