from calculator import divide, parse_and_add


def test_divide():
    assert divide(6, 3) == 2


def test_parse_and_add():
    assert parse_and_add("1", "2") == 3
