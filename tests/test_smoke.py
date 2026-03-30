from __future__ import annotations

from psorad.cli import build_parser


def test_build_parser() -> None:
    parser = build_parser()
    assert parser is not None
