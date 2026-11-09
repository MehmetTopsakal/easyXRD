"""Small provenance records suitable for NetCDF attributes."""
import hashlib
import json
from importlib import metadata
from pathlib import Path
import numpy as np


def file_record(path):
    """Identify a calibration/instrument file by path and content hash."""
    if path is None:
        return None
    path = Path(path).expanduser().resolve()
    with path.open("rb") as stream:
        digest = hashlib.file_digest(stream, "sha256").hexdigest()
    return {"path": str(path), "sha256": digest}


def software_versions(*packages):
    versions = {}
    for name in packages:
        try:
            versions[name] = metadata.version(name)
        except metadata.PackageNotFoundError:
            versions[name] = "unavailable"
    return versions


def provenance_json(record):
    """Encode NumPy scalars/arrays while rejecting unsupported object types."""
    def encode(value):
        if isinstance(value, np.ndarray):
            return value.tolist()
        if isinstance(value, np.generic):
            return value.item()
        raise TypeError(f"Unsupported provenance value: {type(value).__name__}")
    return json.dumps(record, default=encode, sort_keys=True, allow_nan=False)
