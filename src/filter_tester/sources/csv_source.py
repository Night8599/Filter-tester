"""Data source that reads recorded measurements (and optional truth) from CSV files."""

from __future__ import annotations

import csv
import os
from collections.abc import Iterator, Mapping, Sequence
from dataclasses import dataclass
from typing import Union

import numpy as np

from ..core.step import Step

PathLike = Union[str, "os.PathLike[str]"]

TIME_COLUMN_CANDIDATES = ("t", "time", "timestamp")
TIME_UNITS = {"s": 1.0, "ms": 1e-3, "us": 1e-6, "µs": 1e-6, "ns": 1e-9, "min": 60.0, "h": 3600.0}
TRUTH = "truth"
_MERGE_TOL = 1e-9  # seconds; samples closer than this are the same instant


@dataclass
class _Series:
    """One sensor (or the truth) read from one file."""

    name: str
    file: str
    columns: list[str]
    times: np.ndarray  # (k,) seconds, sorted
    values: np.ndarray  # (k, m)


class CSVSource:
    """Reads one CSV file per sensor (and optionally the truth) and yields :class:`Step`\\ s.

    Every file has a header row and uses ``,`` as separator. Times are
    converted to seconds using ``time_unit`` (``"s"``, ``"ms"``, ``"us"``,
    ``"ns"``, ``"min"``, ``"h"``).

    **Automatic mode** (``CSVSource(files)``). In each file:

    - the time column is ``time_column`` if given, otherwise the first of
      ``t``, ``time``, ``timestamp`` found (case-insensitive);
    - a column named ``prefix.column`` belongs to the group ``prefix``;
    - a column without a prefix belongs to the group named after the file
      (``gps.csv`` -> ``gps``);
    - the group ``truth`` is the ground truth, so it can come from its own
      ``truth.csv`` or from ``truth.*`` columns in another file.

    **Explicit mode** (``CSVSource(mapping=...)``), e.g.::

        CSVSource(mapping={
            "gps":   {"file": "gps.csv", "columns": ["x", "y"]},
            "truth": {"file": "log.csv", "columns": ["x_true", "v_true"], "time_column": "stamp"},
        })

    Only the listed groups and columns are read.

    In both modes, a row whose group columns are all empty means that group
    has no value at that time (so several sensors can share one file with
    different rates); a partially empty row is an error. Samples from all
    groups at the same instant form a single step. Steps carry the truth
    only where the truth has a sample.
    """

    def __init__(
        self,
        files: PathLike | Sequence[PathLike] | None = None,
        mapping: Mapping[str, Mapping] | None = None,
        time_column: str | None = None,
        time_unit: str = "s",
    ):
        if (files is None) == (mapping is None):
            raise ValueError("give exactly one of files (automatic mode) or mapping (explicit mode)")
        if time_unit not in TIME_UNITS:
            raise ValueError(f"time_unit must be one of {sorted(TIME_UNITS)}, got {time_unit!r}")
        self.time_unit = time_unit
        self.time_column = time_column
        if files is not None:
            if isinstance(files, (str, os.PathLike)):
                files = [files]
            series = [s for f in files for s in self._read_auto(os.fspath(f))]
        else:
            series = [self._read_mapped(name, spec) for name, spec in mapping.items()]

        names = [s.name for s in series]
        duplicates = sorted({n for n in names if names.count(n) > 1})
        if duplicates:
            raise ValueError(f"groups found in more than one place: {duplicates}")
        self._truth = next((s for s in series if s.name == TRUTH), None)
        self._sensors = {s.name: s for s in series if s.name != TRUTH}

    # ------------------------------------------------------------------ info
    @property
    def sensor_names(self) -> list[str]:
        """Names of the sensors found in the files."""
        return list(self._sensors)

    @property
    def has_truth(self) -> bool:
        """Whether the files contain ground truth."""
        return self._truth is not None

    def columns(self, name: str) -> list[str]:
        """Column names (without prefix) of a sensor, or of ``"truth"``."""
        if name == TRUTH and self._truth is not None:
            return list(self._truth.columns)
        return list(self._sensors[name].columns)

    # ----------------------------------------------------------------- steps
    def steps(self) -> Iterator[Step]:
        """Yield the recorded data one step per instant, in time order."""
        series = list(self._sensors.values()) + ([self._truth] if self._truth else [])
        events = sorted(
            ((t, s.name, k) for s in series for k, t in enumerate(s.times)),
            key=lambda e: e[0],
        )
        i = 0
        while i < len(events):
            t = events[i][0]
            group = [events[i]]
            i += 1
            while i < len(events) and events[i][0] - t <= _MERGE_TOL:
                group.append(events[i])
                i += 1
            measurements, truth = {}, None
            for _, name, k in group:
                if name == TRUTH:
                    truth = self._truth.values[k].copy()
                else:
                    measurements[name] = self._sensors[name].values[k].copy()
            yield Step(t=float(t), measurements=measurements, truth=truth)

    def run(self) -> list[Step]:
        """Return all steps."""
        return list(self.steps())

    # --------------------------------------------------------------- reading
    def _read_auto(self, path: str) -> list[_Series]:
        header, rows = _read_csv(path)
        time_col = self._find_time_column(header, path, self.time_column)
        stem = os.path.splitext(os.path.basename(path))[0]
        groups: dict[str, list[tuple[str, str]]] = {}  # group -> [(column header, short name)]
        for col in header:
            if col == time_col:
                continue
            prefix, sep, rest = col.partition(".")
            group, short = (prefix, rest) if sep and prefix and rest else (stem, col)
            groups.setdefault(group, []).append((col, short))
        if not groups:
            raise ValueError(f"{path}: no data columns besides the time column {time_col!r}")
        return [
            self._series(name, path, header, rows, time_col, [c for c, _ in cols], [s for _, s in cols])
            for name, cols in groups.items()
        ]

    def _read_mapped(self, name: str, spec: Mapping) -> _Series:
        unknown = set(spec) - {"file", "columns", "time_column"}
        if unknown:
            raise ValueError(f"mapping for {name!r}: unknown keys {sorted(unknown)}")
        if "file" not in spec or "columns" not in spec:
            raise ValueError(f"mapping for {name!r} needs 'file' and 'columns'")
        path = os.fspath(spec["file"])
        columns = [spec["columns"]] if isinstance(spec["columns"], str) else list(spec["columns"])
        if not columns:
            raise ValueError(f"mapping for {name!r}: columns must not be empty")
        header, rows = _read_csv(path)
        missing = [c for c in columns if c not in header]
        if missing:
            raise ValueError(f"{path}: columns {missing} not found (available: {header})")
        time_col = self._find_time_column(header, path, spec.get("time_column", self.time_column))
        return self._series(name, path, header, rows, time_col, columns, columns)

    def _series(self, name, path, header, rows, time_col, columns, short_names) -> _Series:
        ti = header.index(time_col)
        ci = [header.index(c) for c in columns]
        times, values = [], []
        for line, row in rows:
            cells = [row[i].strip() if i < len(row) else "" for i in ci]
            filled = [c != "" and c.lower() != "nan" for c in cells]
            if not any(filled):
                continue
            if not all(filled):
                raise ValueError(f"{path}, line {line}: {name!r} is only partially filled")
            t_cell = row[ti].strip() if ti < len(row) else ""
            if t_cell == "":
                raise ValueError(f"{path}, line {line}: empty time")
            times.append(_to_float(t_cell, path, line) * TIME_UNITS[self.time_unit])
            values.append([_to_float(c, path, line) for c in cells])
        if not times:
            raise ValueError(f"{path}: {name!r} has no data")
        times = np.array(times)
        values = np.array(values)
        order = np.argsort(times, kind="stable")
        times, values = times[order], values[order]
        if np.any(np.diff(times) <= _MERGE_TOL):
            raise ValueError(f"{path}: {name!r} has two samples at the same time")
        return _Series(name, path, list(short_names), times, values)

    @staticmethod
    def _find_time_column(header: list[str], path: str, requested: str | None) -> str:
        if requested is not None:
            if requested not in header:
                raise ValueError(f"{path}: time column {requested!r} not found (available: {header})")
            return requested
        lowered = {h.lower(): h for h in header}
        for candidate in TIME_COLUMN_CANDIDATES:
            if candidate in lowered:
                return lowered[candidate]
        raise ValueError(
            f"{path}: no time column found (looked for {list(TIME_COLUMN_CANDIDATES)}); "
            "pass time_column=... to choose one"
        )


def _read_csv(path: str) -> tuple[list[str], list[tuple[int, list[str]]]]:
    with open(path, newline="", encoding="utf-8-sig") as f:
        reader = csv.reader(f, delimiter=",")
        try:
            header = [h.strip() for h in next(reader)]
        except StopIteration:
            raise ValueError(f"{path}: file is empty") from None
        if len(set(header)) != len(header):
            raise ValueError(f"{path}: repeated column names in header")
        rows = [(reader.line_num, row) for row in reader if any(cell.strip() for cell in row)]
    return header, rows


def _to_float(cell: str, path: str, line: int) -> float:
    try:
        return float(cell)
    except ValueError:
        raise ValueError(f"{path}, line {line}: {cell!r} is not a number") from None
