"""Tests for the DICOM to Hounsfield unit conversion in src/ich/data/dicom.py."""
import numpy as np
import pytest
from pydicom.dataset import Dataset, FileMetaDataset
from pydicom.uid import CTImageStorage, ExplicitVRLittleEndian, generate_uid

from ich.data.dicom import HU_MAX, HU_MIN, read_hu, scaled_spacing, to_hu, to_square


def write_dicom(path, pixels, bits_stored=16, signed=True, intercept=-1024, pixel_data=None):
    """A minimal CT file with the given stored values; `pixel_data` replaces the pixel bytes."""
    meta = FileMetaDataset()
    meta.MediaStorageSOPClassUID = CTImageStorage
    meta.MediaStorageSOPInstanceUID = generate_uid()
    meta.TransferSyntaxUID = ExplicitVRLittleEndian
    ds = Dataset()
    ds.file_meta = meta
    ds.SOPClassUID, ds.SOPInstanceUID = meta.MediaStorageSOPClassUID, meta.MediaStorageSOPInstanceUID
    ds.Rows, ds.Columns = pixels.shape
    ds.SamplesPerPixel, ds.PhotometricInterpretation = 1, "MONOCHROME2"
    ds.BitsAllocated, ds.BitsStored, ds.HighBit = 16, bits_stored, bits_stored - 1
    ds.PixelRepresentation = int(signed)
    ds.RescaleIntercept, ds.RescaleSlope = str(intercept), "1"
    ds.PixelData = pixels.astype(np.int16 if signed else np.uint16).tobytes() if pixel_data is None else pixel_data
    ds.save_as(path, enforce_file_format=True)
    return path


def test_values_are_cast_before_the_rescale():
    stored = np.array([0, 24, 4095], dtype=np.uint16)
    assert to_hu(stored, 1, -1024, 16, 0).tolist() == [-1024, -1000, 3071]


@pytest.mark.parametrize("intercept", [0, 1])
def test_intercept_fix_reads_hu_modulo_4096(intercept):
    # HU -1000, -1, 0, 40, 2000, 3095 stored modulo 4096 in a file marked 12 bit unsigned
    stored = np.array([3096, 4095, 0, 40, 2000, 3095], dtype=np.uint16)
    assert to_hu(stored, 1, intercept, 12, 0).tolist() == [-1000, -1, 0, 40, 2000, 3095]


@pytest.mark.parametrize("bits_stored, pixel_representation, intercept", [(12, 0, -1024), (12, 1, 0), (16, 0, 0)])
def test_no_fix_for_other_files(bits_stored, pixel_representation, intercept):
    stored = np.array([1024, 3096], dtype=np.uint16)
    assert to_hu(stored, 1, intercept, bits_stored, pixel_representation).tolist() == [1024 + intercept, 3096 + intercept]


def test_to_hu_does_not_change_its_input():
    stored = np.array([3096.0, 40.0], dtype=np.float32)
    to_hu(stored, 1, 0, 12, 0)
    assert stored.tolist() == [3096, 40]


def test_non_square_image_is_padded_with_air_in_the_centre():
    hu = np.full((4, 2), 100, dtype=np.float32)
    square = to_square(hu, size=4)
    assert square.shape == (4, 4)
    assert (square[:, [0, 3]] == HU_MIN).all() and (square[:, 1:3] == 100).all()


@pytest.mark.parametrize("side", [374, 464, 512, 768])
def test_square_image_is_resized_without_changing_a_constant(side):
    out = to_square(np.full((side, side), 40, dtype=np.float32))
    assert out.shape == (512, 512) and out.dtype == np.float32
    assert np.allclose(out, 40)


def test_scaled_spacing_follows_the_longer_side():
    assert scaled_spacing(0.5, 638, 490) == pytest.approx(0.5 * 638 / 512)
    assert scaled_spacing(0.488, 512, 512) == pytest.approx(0.488)


def test_read_hu_on_a_mislabelled_non_square_file(tmp_path):
    stored = np.full((600, 400), 3096, dtype=np.uint16)  # air, stored modulo 4096
    stored[200:400, 100:300] = 40                         # brain
    hu = read_hu(write_dicom(tmp_path / "a.dcm", stored, bits_stored=12, signed=False, intercept=0))
    assert hu.shape == (512, 512) and hu.dtype == np.float32
    assert hu.min() >= HU_MIN and hu.max() <= HU_MAX
    assert hu[0, 0] == HU_MIN                   # padding on the left and right
    assert hu[0, 256] == pytest.approx(-1000)   # air, after the fix
    assert hu[256, 256] == pytest.approx(40)    # brain, unchanged


def test_read_hu_clips_extreme_values(tmp_path):
    stored = np.array([[-32768, 0], [100, 32767]], dtype=np.int16)
    hu = read_hu(write_dicom(tmp_path / "b.dcm", stored, intercept=0), size=2)
    assert hu.tolist() == [[HU_MIN, 0], [100, HU_MAX]]


def test_read_hu_raises_on_missing_pixel_data(tmp_path):
    stored = np.zeros((64, 64), dtype=np.int16)
    path = write_dicom(tmp_path / "c.dcm", stored, pixel_data=stored.tobytes()[:3000])
    with pytest.raises(ValueError):
        read_hu(path)
