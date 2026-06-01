"""Pytest configuration: force OFFLINE mode so tests never open a socket."""

import os

os.environ.setdefault("OPERATION_MODE", "offline")
