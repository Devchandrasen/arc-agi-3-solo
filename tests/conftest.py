"""Pytest configuration: force offline (competition) mode so tests don't need internet."""

import os

os.environ.setdefault("OPERATION_MODE", "competition")
