import pytest

import flcore.utils.atomic_io as atomic_io


def test_dump_success(tmp_path):
    atomic_io.dump(obj=1, filename=tmp_path / "test_dump_success.pt", replace=True)


def test_dump_fail_file_exists(tmp_path):
    filename = tmp_path / "test_dump_fail_file_exists.pt"
    filename.touch()

    with pytest.raises(FileExistsError):
        atomic_io.dump(obj=1, filename=filename, replace=False)


def test_dump_atomicity(tmp_path):
    filename = tmp_path / "test_dump_atomicity.pt"
    temp_filename = filename.with_suffix(".tmp")
    temp_filename.touch()

    atomic_io.dump(obj=1, filename=filename, replace=True)

    assert temp_filename.exists() is False


def test_load_success(tmp_path):
    filename = tmp_path / "test_load_success.pt"
    atomic_io.dump(obj=1, filename=filename, replace=True)

    assert atomic_io.load(filename=filename) == 1


def test_load_fail_file_not_exists(tmp_path):
    filename = tmp_path / "test_load_fail_file_not_exists.pt"

    with pytest.raises(FileNotFoundError):
        atomic_io.load(filename=filename)
