#!/usr/bin/env python3
import os
from pathlib import Path

from snakebids import bidsapp, plugins

if "__file__" not in globals():
    __file__ = "../nnunet_contact_seg/run.py"

app = bidsapp.app(
    [
        plugins.SnakemakeBidsApp(Path(__file__).resolve().parent),
        plugins.BidsValidator(),
        plugins.Version(distribution="nnUNet_contact_seg"),
        plugins.CliConfig("parse_args"),
        plugins.ComponentEdit("pybids_inputs"),
    ]
)


def get_parser():
    """Exposes parser for sphinx doc generation, cwd is the docs dir."""
    return app.build_parser().parser


if __name__ == "__main__":
    app.run()
