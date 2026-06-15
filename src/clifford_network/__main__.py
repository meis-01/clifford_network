"""Module entry point for `python -m clifford_network`.

It delegates directly to the CLI dispatcher so package execution behaves the
same as the installed `clifford-network` console script.
"""

from clifford_network.cli import main


if __name__ == "__main__":
    main()
