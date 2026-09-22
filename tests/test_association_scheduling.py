"""Association scheduling regressions; run with the GaMMA root on PYTHONPATH."""

from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
from threadpoolctl import threadpool_info, threadpool_limits

from gamma import utils


@pytest.fixture
def association_case(monkeypatch):
    case = SimpleNamespace(dispatched=[], pools=[], contexts=[], empty_clusters=set())

    def prepare(labels, *, ncpu=2, use_dbscan=True):
        labels = np.asarray(labels)
        count = len(labels)
        monkeypatch.setattr(
            utils,
            "convert_picks_csv",
            lambda *_args: (
                np.arange(count, dtype=float)[:, None],
                np.zeros((count, 3)),
                np.full(count, "p"),
                np.ones((count, 1)),
                np.arange(count),
                np.array([f"station_{i}_p" for i in range(count)]),
                0.0,
            ),
        )
        monkeypatch.setattr(utils, "hierarchical_dbscan_clustering", lambda *_args, **_kwargs: labels)
        return {
            "ncpu": ncpu,
            "min_picks_per_eq": 1,
            "use_dbscan": use_dbscan,
            "dbscan_eps": 10.0,
            "dbscan_min_samples": 1,
            "eikonal": None,
        }

    def associate(label, *_args):
        case.dispatched.append(label)
        if label in case.empty_clusters:
            return [], []
        return (
            [
                {"event_index": 3, "cluster_label": int(label), "gamma_score": 0.9},
                {"event_index": 8, "cluster_label": int(label), "gamma_score": 0.8},
            ],
            [(int(label) * 10, 3, 0.9), (int(label) * 10 + 1, 8, 0.8)],
        )

    class Pool:
        def __init__(self, processes, initializer):
            case.pools.append((processes, initializer))

        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return None

        def starmap(self, function, tasks, chunksize):
            assert chunksize == 1
            return [function(*task) for task in tasks]

    def get_context(method):
        case.contexts.append(method)
        return SimpleNamespace(Pool=Pool)

    monkeypatch.setattr(utils, "associate", associate)
    monkeypatch.setattr(utils.mp, "get_context", get_context)
    monkeypatch.setattr(utils.os, "sched_getaffinity", lambda _pid: set(range(8)), raising=False)
    case.prepare = prepare
    return case


@pytest.mark.parametrize("ncpu", [1, 2])
@pytest.mark.parametrize(
    "labels, expected_dispatch, expected_merge",
    [
        ([0, 2, 2, 2, 1, 1, -1], [2, 1, 0], [0, 1, 2]),
        ([2, 0, 2, 0], [0, 2], [0, 2]),
        ([4, 4, 4], [4], [4]),
        ([-1, -1], [], []),
    ],
)
def test_scheduling_preserves_label_ordered_outputs(
    association_case, ncpu, labels, expected_dispatch, expected_merge
):
    config = association_case.prepare(labels, ncpu=ncpu)

    events, assignments = utils.association(None, None, config, event_idx0=7)

    assert association_case.dispatched == expected_dispatch
    expected_events = []
    expected_assignments = []
    for position, label in enumerate(expected_merge):
        for offset, score in enumerate((0.9, 0.8)):
            event_id = 7 + 2 * position + offset
            expected_events.append(
                {"event_index": event_id, "cluster_label": label, "gamma_score": score}
            )
            expected_assignments.append((label * 10 + offset, event_id, score))
    assert events == expected_events
    assert assignments == expected_assignments
    if ncpu == 1 or len(expected_dispatch) < 2:
        assert association_case.pools == []
    else:
        assert association_case.pools == [(2, utils._init_association_worker)]


@pytest.mark.parametrize("ncpu", [1, 2])
def test_empty_cluster_results_do_not_leave_event_id_gaps(association_case, ncpu):
    config = association_case.prepare([0, 1, 1, 1, 2, 2], ncpu=ncpu)
    association_case.empty_clusters.add(1)

    events, assignments = utils.association(None, None, config, event_idx0=7)

    assert association_case.dispatched == [1, 2, 0]
    assert [event["cluster_label"] for event in events] == [0, 0, 2, 2]
    assert [event["event_index"] for event in events] == [7, 8, 9, 10]
    assert assignments == [(0, 7, 0.9), (1, 8, 0.8), (20, 9, 0.9), (21, 10, 0.8)]


def test_dbscan_disabled_uses_one_serial_cluster(association_case, monkeypatch):
    config = association_case.prepare([9, 8, 7], ncpu=8, use_dbscan=False)

    def unexpected_dbscan(*_args, **_kwargs):
        pytest.fail("DBSCAN must not run when disabled")

    monkeypatch.setattr(utils, "hierarchical_dbscan_clustering", unexpected_dbscan)
    events, _assignments = utils.association(None, None, config)

    assert association_case.dispatched == [0]
    assert association_case.pools == []
    assert [event["event_index"] for event in events] == [0, 1]


@pytest.mark.parametrize("raises", [False, True])
def test_serial_thread_limits_are_restored(association_case, monkeypatch, raises):
    config = association_case.prepare([0, 0], ncpu=1)

    def associate(*_args):
        assert all(pool["num_threads"] == 1 for pool in threadpool_info())
        if raises:
            raise RuntimeError("association failed")
        return [], []

    monkeypatch.setattr(utils, "associate", associate)
    with threadpool_limits(limits=2):
        before = threadpool_info()
        assert before, "The test requires a loaded native thread pool"
        if raises:
            with pytest.raises(RuntimeError, match="association failed"):
                utils.association(None, None, config)
        else:
            assert utils.association(None, None, config) == ([], [])
        assert threadpool_info() == before
    assert association_case.pools == []


@pytest.mark.parametrize(
    "system, torch_loaded, expected_context",
    [
        ("Linux", False, "fork"),
        ("Linux", True, "spawn"),
        ("Darwin", False, "spawn"),
        ("Windows", False, "spawn"),
    ],
)
def test_parallel_start_method(
    association_case, monkeypatch, system, torch_loaded, expected_context
):
    config = association_case.prepare([0, 0, 1, 1])
    monkeypatch.setattr(utils.platform, "system", lambda: system)
    if torch_loaded:
        monkeypatch.setitem(utils.sys.modules, "torch", SimpleNamespace())
    else:
        monkeypatch.delitem(utils.sys.modules, "torch", raising=False)

    utils.association(None, None, config)

    assert association_case.contexts == [expected_context]
    assert association_case.pools == [(2, utils._init_association_worker)]


def test_worker_count_respects_cpu_affinity_and_cluster_count(monkeypatch):
    monkeypatch.setattr(utils.os, "sched_getaffinity", lambda _pid: {0, 1, 2}, raising=False)

    assert utils._effective_workers(12, 20) == 3
    assert utils._effective_workers(12, 2) == 2
    assert utils._effective_workers(1, 20) == 1


@pytest.fixture
def synthetic_catalog():
    stations = pd.DataFrame(
        {
            "id": [f"station_{i}" for i in range(6)],
            "x(km)": [-10.0, 10.0, -10.0, 10.0, 0.0, 0.0],
            "y(km)": [-10.0, -10.0, 10.0, 10.0, -15.0, 15.0],
            "z(km)": np.zeros(6),
        }
    )
    config = {
        "dims": ["x(km)", "y(km)", "z(km)"],
        "z(km)": (0, 20),
        "vel": {"p": 6.0, "s": 3.5},
        "use_amplitude": False,
        "use_dbscan": True,
        "dbscan_eps": 10.0,
        "dbscan_min_samples": 3,
        "min_picks_per_eq": 4,
        "oversample_factor": 1,
        "covariance_prior": [1.0, 1.0],
        "max_sigma11": 1.0,
        "eikonal": None,
        "bfgs_bounds": ((-20, 20), (-20, 20), (0, 20), (None, None)),
    }
    picks = []
    # The later event has more picks, so it must be scheduled first.
    for origin_time, location, station_count in [(0, [-1, 0, 6], 4), (120, [1, 2, 8], 6)]:
        for _, station in stations.head(station_count).iterrows():
            distance = np.linalg.norm(station[config["dims"]].to_numpy(dtype=float) - location)
            for phase, velocity in config["vel"].items():
                picks.append(
                    {
                        "id": station["id"],
                        "type": phase,
                        "prob": 0.99,
                        "timestamp": pd.Timestamp("2020-01-01", tz="UTC")
                        + pd.Timedelta(seconds=origin_time + distance / velocity),
                    }
                )
    return pd.DataFrame(picks), stations, config


def test_real_spawn_workers_match_serial_results(synthetic_catalog, monkeypatch):
    picks, stations, config = synthetic_catalog
    get_context = utils.mp.get_context
    monkeypatch.setattr(utils, "mp", SimpleNamespace(get_context=lambda _method: get_context("spawn")))
    monkeypatch.setattr(utils.os, "sched_getaffinity", lambda _pid: {0, 1}, raising=False)

    serial = utils.association(picks.copy(), stations, {**config, "ncpu": 1}, event_idx0=7)
    parallel = utils.association(picks.copy(), stations, {**config, "ncpu": 2}, event_idx0=7)

    assert parallel == serial
    assert [event["event_index"] for event in serial[0]] == [7, 8]
    assert len(serial[1]) == len(picks)
