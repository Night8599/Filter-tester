import numpy as np
import pytest

from filter_tester.core import measurement_series, truth_series
from filter_tester.sources import CSVSource


def write(path, text):
    path.write_text(text.strip() + "\n")
    return path


# ---------------------------------------------------------------- automatic
def test_one_file_per_sensor_named_after_file(tmp_path):
    gps = write(tmp_path / "gps.csv", "t,x,y\n0.0,1,2\n0.5,3,4\n1.0,5,6")
    imu = write(tmp_path / "imu.csv", "time,ax\n0.0,0.1\n0.25,0.2\n0.5,0.3")
    source = CSVSource([gps, imu])
    assert source.sensor_names == ["gps", "imu"]
    assert not source.has_truth
    assert source.columns("gps") == ["x", "y"]
    steps = source.run()
    assert [s.t for s in steps] == [0.0, 0.25, 0.5, 1.0]
    assert set(steps[0].measurements) == {"gps", "imu"}  # same instant -> one step
    assert set(steps[1].measurements) == {"imu"}
    assert np.array_equal(steps[2].measurements["gps"], [3.0, 4.0])
    assert all(s.truth is None for s in steps)


def test_truth_in_its_own_file(tmp_path):
    gps = write(tmp_path / "gps.csv", "t,x\n0.0,1.1\n1.0,2.1")
    truth = write(tmp_path / "truth.csv", "t,x,v\n0.0,1,1\n0.5,1.5,1\n1.0,2,1")
    source = CSVSource([gps, truth])
    assert source.has_truth and source.sensor_names == ["gps"]
    assert source.columns("truth") == ["x", "v"]
    t, x = truth_series(source.run())
    assert np.allclose(t, [0.0, 0.5, 1.0]) and x.shape == (3, 2)


def test_prefixed_columns_in_one_file(tmp_path):
    log = write(
        tmp_path / "log.csv",
        """
        timestamp,gps.x,imu.ax,truth.x,truth.v
        0.0,1.0,0.1,1,1
        0.1,,0.2,1.1,1
        0.2,1.2,0.3,1.2,1
        """.replace("        ", ""),
    )
    source = CSVSource(log)
    assert sorted(source.sensor_names) == ["gps", "imu"]
    steps = source.run()
    assert len(steps) == 3
    assert set(steps[1].measurements) == {"imu"}  # empty gps cell -> no gps measurement
    assert all(s.truth is not None for s in steps)
    tz, z = measurement_series(steps, "gps")
    assert np.allclose(tz, [0.0, 0.2]) and np.allclose(z[:, 0], [1.0, 1.2])


def test_unprefixed_columns_go_to_file_group(tmp_path):
    f = write(tmp_path / "gps.csv", "t,x,truth.x\n0,1.1,1\n1,2.1,2")
    source = CSVSource(f)
    assert source.sensor_names == ["gps"] and source.has_truth


def test_time_column_detection_is_case_insensitive(tmp_path):
    f = write(tmp_path / "s.csv", "Time,a\n0,1\n1,2")
    assert [s.t for s in CSVSource(f).run()] == [0.0, 1.0]


def test_explicit_time_column(tmp_path):
    f = write(tmp_path / "s.csv", "stamp,a\n0,1\n1,2")
    with pytest.raises(ValueError, match="no time column"):
        CSVSource(f)
    assert [s.t for s in CSVSource(f, time_column="stamp").run()] == [0.0, 1.0]


@pytest.mark.parametrize("unit, factor", [("s", 1.0), ("ms", 1e-3), ("us", 1e-6), ("min", 60.0)])
def test_time_unit_conversion(tmp_path, unit, factor):
    f = write(tmp_path / "s.csv", "t,a\n0,1\n250,2")
    assert np.allclose([s.t for s in CSVSource(f, time_unit=unit).run()], [0.0, 250 * factor])


def test_unsorted_rows_are_sorted(tmp_path):
    f = write(tmp_path / "s.csv", "t,a\n1,2\n0,1")
    steps = CSVSource(f).run()
    assert [s.t for s in steps] == [0.0, 1.0]
    assert steps[0].measurements["s"][0] == 1.0


def test_nan_cells_count_as_empty(tmp_path):
    f = write(tmp_path / "log.csv", "t,a.x,b.x\n0,1,NaN\n1,nan,2")
    steps = CSVSource(f).run()
    assert [set(s.measurements) for s in steps] == [{"a"}, {"b"}]


# ----------------------------------------------------------------- explicit
def test_explicit_mapping(tmp_path):
    f = write(tmp_path / "log.csv", "t,x_meas,y_meas,x_true,junk\n0,1,2,1,9\n1,3,4,3,9")
    g = write(tmp_path / "other.csv", "stamp,speed\n0.5,1.0")
    source = CSVSource(
        mapping={
            "gps": {"file": f, "columns": ["x_meas", "y_meas"]},
            "truth": {"file": f, "columns": "x_true"},
            "speedometer": {"file": g, "columns": ["speed"], "time_column": "stamp"},
        }
    )
    assert sorted(source.sensor_names) == ["gps", "speedometer"]
    assert source.columns("gps") == ["x_meas", "y_meas"]
    steps = source.run()
    assert [s.t for s in steps] == [0.0, 0.5, 1.0]
    assert steps[1].truth is None and steps[1].measurements["speedometer"][0] == 1.0
    assert np.array_equal(steps[2].truth, [3.0])


@pytest.mark.parametrize(
    "spec",
    [
        {"columns": ["a"]},
        {"file": None, "columns": []},
        {"file": None, "columns": ["missing"]},
        {"file": None, "columns": ["a"], "extra": 1},
    ],
)
def test_explicit_mapping_errors(tmp_path, spec):
    f = write(tmp_path / "s.csv", "t,a\n0,1")
    spec = {k: (f if k == "file" else v) for k, v in spec.items()}
    with pytest.raises(ValueError):
        CSVSource(mapping={"s": spec})


# ------------------------------------------------------------------- errors
def test_requires_exactly_one_mode(tmp_path):
    f = write(tmp_path / "s.csv", "t,a\n0,1")
    with pytest.raises(ValueError):
        CSVSource()
    with pytest.raises(ValueError):
        CSVSource(f, mapping={"s": {"file": f, "columns": ["a"]}})


@pytest.mark.parametrize(
    "text, match",
    [
        ("t,a,b\n0,1,", "partially filled"),
        ("t,a\n0,1\n0,2", "same time"),
        ("t,a\n0,abc", "not a number"),
        ("t,a\n,1", "empty time"),
        ("t,a,a\n0,1,2", "repeated column"),
        ("t\n0", "no data columns"),
    ],
)
def test_bad_files(tmp_path, text, match):
    with pytest.raises(ValueError, match=match):
        CSVSource(write(tmp_path / "s.csv", text))


def test_same_group_in_two_files_rejected(tmp_path):
    a = write(tmp_path / "a.csv", "t,gps.x\n0,1")
    b = write(tmp_path / "b.csv", "t,gps.x\n1,2")
    with pytest.raises(ValueError, match="more than one place"):
        CSVSource([a, b])


def test_bad_time_unit(tmp_path):
    with pytest.raises(ValueError):
        CSVSource(write(tmp_path / "s.csv", "t,a\n0,1"), time_unit="days")
