import pytest

from cli import parse_args


def test_parse_args_defaults():
    args = parse_args([])
    assert "hello.py" in args.goal
    assert args.plan_steps == 10
    assert args.build_steps == 20
    assert args.quiet is False
    assert args.persist is True


def test_parse_args_goal_and_flags():
    args = parse_args(
        [
            "fix the bug in main.py",
            "--plan-steps",
            "3",
            "--build-steps",
            "5",
            "-q",
            "--no-persist",
        ]
    )
    assert args.goal == "fix the bug in main.py"
    assert args.plan_steps == 3
    assert args.build_steps == 5
    assert args.quiet is True
    assert args.persist is False


def test_parse_args_rejects_bad_int():
    with pytest.raises(SystemExit):
        parse_args(["--plan-steps", "x"])
