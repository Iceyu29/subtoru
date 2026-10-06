"""Exporters initialization. Imports all exporter modules to trigger registration."""

from __future__ import annotations

import contextlib

from subtoru.exporters import pysubs2_formats as pysubs2_formats  # noqa: F401

with contextlib.suppress(ImportError):
    from subtoru.exporters import lrc as lrc  # noqa: F401

with contextlib.suppress(ImportError):
    from subtoru.exporters import json_export as json_export  # noqa: F401

with contextlib.suppress(ImportError):
    from subtoru.exporters import txt as txt  # noqa: F401

with contextlib.suppress(ImportError):
    from subtoru.exporters import csv_export as csv_export  # noqa: F401

with contextlib.suppress(ImportError):
    from subtoru.exporters import sbv as sbv  # noqa: F401

with contextlib.suppress(ImportError):
    from subtoru.exporters import bilingual as bilingual  # noqa: F401
