"""Read what a model can do from its GGUF metadata, as llama.cpp does.

llama.cpp marks decision models (POST /v1/systemone) with an
"<arch>.decision.type" key; see common_get_gguf_info() in common/common.cpp
and server_model_output_modalities() in tools/server/server-common.cpp.
"""

import struct
import zipfile
from dataclasses import dataclass
from typing import BinaryIO

# GGUF value types: fixed sizes, plus STRING (8) and ARRAY (9)
_SIZES = {0: 1, 1: 1, 2: 2, 3: 2, 4: 4, 5: 4, 6: 4, 7: 1, 10: 8, 11: 8, 12: 8}
_STRING, _ARRAY = 8, 9


@dataclass(frozen=True)
class ModelOutputs:
    """The "output_modalities" the server reports for a model."""

    text: bool = True  # chat and text generation
    decisions: bool = False  # POST /v1/systemone
    decision_type: str | None = None


def _read(f: BinaryIO, fmt: str):
    size = struct.calcsize(fmt)
    data = f.read(size)
    if len(data) != size:
        raise ValueError("truncated GGUF")
    return struct.unpack(fmt, data)[0]


def _string(f: BinaryIO) -> str:
    return f.read(_read(f, "<Q")).decode("utf-8", errors="replace")


def _skip(f: BinaryIO, vtype: int) -> None:
    if vtype == _STRING:
        f.seek(_read(f, "<Q"), 1)
    elif vtype == _ARRAY:
        etype, count = _read(f, "<I"), _read(f, "<Q")
        if etype in _SIZES:
            f.seek(_SIZES[etype] * count, 1)
        else:
            for _ in range(count):
                _skip(f, etype)
    elif vtype in _SIZES:
        f.seek(_SIZES[vtype], 1)
    else:
        raise ValueError(f"unknown GGUF value type {vtype}")


def read_decision_type(f: BinaryIO) -> str | None:
    """The "<arch>.decision.type" string of a GGUF stream, or None."""
    if f.read(4) != b"GGUF":
        raise ValueError("not a GGUF file")
    _read(f, "<I")  # version
    _read(f, "<Q")  # tensor count
    strings = {}
    for _ in range(_read(f, "<Q")):
        key, vtype = _string(f), _read(f, "<I")
        if vtype == _STRING and (key == "general.architecture" or key.endswith(".decision.type")):
            strings[key] = _string(f)
        else:
            _skip(f, vtype)
    arch = strings.get("general.architecture")
    return strings.get(f"{arch}.decision.type") if arch else None


def _bundled_gguf(path: str):
    """The first .gguf stored in a llamafile bundle (a zip), or None."""
    try:
        zf = zipfile.ZipFile(path)
    except (OSError, zipfile.BadZipFile):
        return None
    names = [n for n in zf.namelist() if n.endswith(".gguf") and "mmproj" not in n]
    return zf.open(names[0]) if names else None


def model_outputs(model: str | None, executable: str | None = None) -> ModelOutputs:
    """What the model given with --model (or bundled in the executable) can do.

    Mirrors server_model_output_modalities(): a model with a decision type
    serves /v1/systemone and, except granite-switch, nothing else. Anything
    that can't be read counts as a text model, as in llama.cpp.
    """
    try:
        f = open(model, "rb") if model else (_bundled_gguf(executable) if executable else None)
        if f is None:
            return ModelOutputs()
        with f:
            decision_type = read_decision_type(f)
    except (OSError, ValueError, KeyError):
        return ModelOutputs()
    if not decision_type:
        return ModelOutputs()
    return ModelOutputs(
        text=decision_type == "granite-switch",
        decisions=True,
        decision_type=decision_type,
    )
