"""Run codec commands; for example: python -m codec encode --help."""

import argparse
from importlib import import_module
import sys

from .utils.legacy_imports import COMMANDS


def build_parser():
    parser = argparse.ArgumentParser(
        prog="python -m codec",
        description="Train quality models and losslessly encode or decode FASTQ files.",
        epilog="Use 'python -m codec COMMAND --help' for command-specific options.")
    parser.add_argument("command", nargs="?", choices=tuple(COMMANDS),
                        help="operation to perform")
    return parser


def main(argv=None):
    arguments = list(sys.argv[1:] if argv is None else argv)
    parser = build_parser()
    if not arguments:
        parser.print_help()
        return 0
    if arguments[0] in ("-h", "--help"):
        parser.parse_args(arguments)
        return 0
    command, remaining = arguments[0], arguments[1:]
    if command not in COMMANDS:
        parser.error("unknown command %r; choose from %s" % (
            command, ", ".join(COMMANDS)))
    implementation = import_module(COMMANDS[command])
    original_arguments = sys.argv
    try:
        sys.argv = ["python -m codec " + command] + remaining
        return implementation.main()
    finally:
        sys.argv = original_arguments


if __name__ == "__main__":
    raise SystemExit(main())
