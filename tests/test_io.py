import pytest

import flcore.utils.io as io


def test_dump_success(tmp_path):
    io.dump(obj=1, filename=tmp_path / "test_dump_success.pt", replace=True)


def test_dump_fail_file_exists(tmp_path):
    filename = tmp_path / "test_dump_fail_file_exists.pt"
    filename.touch()

    with pytest.raises(FileExistsError):
        io.dump(obj=1, filename=filename, replace=False)


def test_dump_atomicity(tmp_path):
    filename = tmp_path / "test_dump_atomicity.pt"
    temp_filename = filename.with_suffix(".tmp")
    temp_filename.touch()

    io.dump(obj=1, filename=filename, replace=True)

    assert temp_filename.exists() is False


def test_load_success(tmp_path):
    filename = tmp_path / "test_load_success.pt"
    io.dump(obj=1, filename=filename, replace=True)

    assert io.load(filename=filename) == 1


def test_load_fail_file_not_exists(tmp_path):
    filename = tmp_path / "test_load_fail_file_not_exists.pt"

    with pytest.raises(FileNotFoundError):
        io.load(filename=filename)
