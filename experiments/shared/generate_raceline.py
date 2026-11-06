"""Entry point for the raceline QP smoother.

Runs ``buzzracer.scripts.qp_smooth`` after this package has installed the
project import paths (the stage does not invoke a module under ``experiments``
directly).
"""

from runpy import run_module


def main():
    run_module("buzzracer.scripts.qp_smooth", run_name="__main__", alter_sys=True)


if __name__ == "__main__":
    main()
