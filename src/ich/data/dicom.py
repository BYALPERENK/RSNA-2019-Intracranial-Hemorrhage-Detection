"""Reading the DICOM files."""
import os

import pydicom
from pydicom.multival import MultiValue

# The image IDs (`ID_...`) are not valid DICOM UIDs; without this pydicom warns on every file.
pydicom.config.settings.reading_validation_mode = pydicom.config.IGNORE


def as_text(value) -> str:
    """A header value as text; multi-valued fields are joined with a backslash, as stored in DICOM."""
    if isinstance(value, MultiValue):
        return "\\".join(str(v) for v in value)
    return str(value)


def read_header(path: str) -> dict:
    """Every header field of one file as text (file meta included, pixels not read), plus the file size.

    A file that cannot be read gets a `read_error` instead of its fields, so no file is lost silently.
    """
    row = {"file": os.path.basename(path), "file_size": os.path.getsize(path)}
    try:
        ds = pydicom.dcmread(path, stop_before_pixels=True)
    except Exception as e:
        row["read_error"] = repr(e)
        return row
    for elem in [*ds.file_meta, *ds]:
        if elem.keyword:
            row[elem.keyword] = as_text(elem.value)
    return row
