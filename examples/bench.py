import time
from pathlib import Path

from session_py import Mesh
from session_py import Point
from session_py import Session
from session_py import Xform
from session_py.history import CAPACITY
from session_py.session import PURGE_WORK

LARGEST = 1_000_000  # The largest scene; the others are a tenth and a thousandth.
SCALE = 40.0  # Slack on bulk and slice budgets for a slower kernel.
WARMUP = 5  # Untimed runs before the timed ones.
RUNS = 101  # Timed runs; their median is the cost.
PER_OBJECT = 0.005  # Milliseconds each object of a bulk step may cost.


def unthrottled() -> bool:
    """Return whether cpu0 runs at its full clock, so absolute budgets mean something."""

    folder = Path("/sys/devices/system/cpu/cpu0/cpufreq")

    try:
        return (folder / "scaling_max_freq").read_text().strip() == (
            folder / "cpuinfo_max_freq"
        ).read_text().strip()
    except OSError:
        return False


def clock(f) -> float:
    """Return the milliseconds one call of f takes."""

    start = time.perf_counter()
    f()

    return (time.perf_counter() - start) * 1e3


def median(times: list[float]) -> float:
    """Return the median of the timings."""
    return sorted(times)[len(times) // 2]


def verdict(name: str, passed: bool) -> None:
    """Print one budget line."""
    print(f"{name:<20} {'ok' if passed else 'OVER BUDGET'}")


def flat(n: int):
    """A session with n points in one flat group, the group and their guids."""

    session = Session()
    group = session.add_group("flat")
    guids = []

    for i in range(n):
        guids.append(session.add_point(Point(float(i), 0.0, 0.0), group).name)

    return session, group, guids


def grid_mesh(n: int) -> Mesh:
    """A quad grid mesh of n by n vertices."""

    vertices = []
    faces = []

    for at in range(n * n):
        vertices.append(Point(float(at // n), float(at % n), 0.0))

    for cell in range((n - 1) * (n - 1)):
        at = cell // (n - 1) * n + cell % (n - 1)
        faces.append([at, at + n, at + n + 1, at + 1])

    return Mesh.from_vertices_and_faces(vertices, faces)


def transaction(session: Session, label: str, edit) -> None:
    """Run edit inside one committed transaction."""

    session.begin(label)
    edit()
    session.commit()


def edit_latency(sizes: list[int]) -> None:
    """Median of remove, undo, redo, replace, move and add for each flat size: must not grow with the size."""

    medians = []

    for n in sizes:
        session, group, guids = flat(n)
        times = [[] for _ in range(6)]

        for run in range(WARMUP + RUNS):
            guid = guids[run]
            other = guids[n - 1 - run]
            point = Point(float(run), 1.0, 0.0)
            lap = [
                clock(
                    lambda: transaction(
                        session, "remove", lambda: session.remove_object(guid)
                    )
                ),
                clock(session.undo),
                clock(session.redo),
                clock(
                    lambda: transaction(
                        session, "replace", lambda: session.replace(other, point)
                    )
                ),
                clock(
                    lambda: transaction(
                        session,
                        "move",
                        lambda: session.set_xform(
                            other, Xform.translation(float(run), 0.0, 0.0)
                        ),
                    )
                ),
                clock(
                    lambda: transaction(
                        session,
                        "add",
                        lambda: session.add_point(Point(float(run), 2.0, 0.0), group),
                    )
                ),
            ]

            if run >= WARMUP:
                for k, lapse in enumerate(lap):
                    times[k].append(lapse)

        row = [median(laps) for laps in times]
        print(
            f"edit latency n={n:<8} remove {row[0]:.4f} undo {row[1]:.4f} redo {row[2]:.4f} replace {row[3]:.4f} move {row[4]:.4f} add {row[5]:.4f} ms"
        )
        medians.append(row)

    fast = all(lapse < 1.0 for row in medians for lapse in row)
    level = all(medians[2][k] < 3.0 * medians[0][k] + 0.05 for k in range(6))
    verdict("edit latency fast", not unthrottled() or fast)
    verdict("edit latency level", level)


def bulk_undo(largest: int, bulk: int) -> None:
    """A bulk remove of bulk objects with its undo and redo: under PER_OBJECT each, level across sizes."""

    medians = []

    for n in [largest, largest // 5]:
        session, _, guids = flat(n)
        times = [[] for _ in range(3)]

        def remove():
            for guid in guids[:bulk]:
                session.remove_object(guid)

        for run in range(WARMUP + RUNS):
            lap = [
                clock(lambda: transaction(session, "remove", remove)),
                clock(session.undo),
                clock(session.redo),
            ]
            session.undo()

            if run >= WARMUP:
                for k, lapse in enumerate(lap):
                    times[k].append(lapse)

        row = [median(laps) for laps in times]
        print(
            f"bulk undo n={n:<8} remove {row[0]:.2f} undo {row[1]:.2f} redo {row[2]:.2f} ms for {bulk} objects"
        )
        medians.append(row)

    budget = PER_OBJECT * bulk * SCALE
    over = all(lapse < budget for lapse in medians[0])
    level = all(medians[0][k] < 1.5 * medians[1][k] for k in range(3))
    print(f"bulk budget {budget:.1f} ms")
    verdict("bulk undo budget", not unthrottled() or over)
    verdict("bulk undo level", level)


def no_pauses(n: int, bulk: int) -> None:
    """Purge and checkpoint slices after a dropped bulk remove: every slice under 16 ms times SCALE."""

    session, _, guids = flat(n)
    session.begin("remove")

    for guid in guids[:bulk]:
        session.remove_object(guid)

    session.commit()

    for step in range(CAPACITY):
        transaction(
            session,
            "move",
            lambda: session.set_xform(
                guids[n - 1], Xform.translation(float(step), 0.0, 0.0)
            ),
        )

    purges = []
    purging = True

    while purging:
        start = time.perf_counter()
        purging = session.purge_step(PURGE_WORK)
        purges.append((time.perf_counter() - start) * 1e3)

    writes = []
    data = None

    while data is None:
        start = time.perf_counter()
        data = session.checkpoint(PURGE_WORK)
        writes.append((time.perf_counter() - start) * 1e3)

    sliced = all(lapse < 16.0 * SCALE for lapse in purges + writes)
    print(
        f"no pauses n={n:<8} {len(purges)} purge slices median {median(purges):.3f} ms max {max(purges):.3f} ms, {len(writes)} write slices max {max(writes):.3f} ms"
    )
    verdict("slice budget", not unthrottled() or sliced)
    verdict("purge slice median", not unthrottled() or median(purges) < 2.0 * SCALE)


def steady_state(n: int) -> None:
    """Edit and undo cost after 10k remove/undo/redo/add cycles with idle purging: level with the first hundred."""

    cycles = min(10_000, n)
    session, group, guids = flat(n)
    edits = []
    undos = []

    for cycle, guid in enumerate(guids[:cycles]):
        edits.append(
            clock(
                lambda: transaction(
                    session, "remove", lambda: session.remove_object(guid)
                )
            )
        )
        undos.append(clock(session.undo))
        session.redo()
        transaction(
            session,
            "add",
            lambda: session.add_point(Point(float(cycle), 1.0, 0.0), group),
        )
        session.purge_step(PURGE_WORK)

    late = cycles - 100
    print(
        f"steady state n={n:<8} edit {median(edits[:100]):.4f} -> {median(edits[late:]):.4f} ms, undo {median(undos[:100]):.4f} -> {median(undos[late:]):.4f} ms, {session.number_of_dead()} dead, {session.history.bytes} bytes"
    )
    verdict(
        "steady edit level", median(edits[late:]) <= 1.5 * median(edits[:100]) + 0.05
    )
    verdict(
        "steady undo level", median(undos[late:]) <= 1.5 * median(undos[:100]) + 0.05
    )


def history_memory() -> None:
    """Commit cost of 200 mesh removes under an 8 MiB budget: level once the budget evicts."""

    session = Session()
    session.history.budget = 8 << 20
    guids = []

    for _ in range(200):
        mesh = grid_mesh(100)
        guids.append(mesh.guid)
        session.add_mesh(mesh)

    commits = []

    for guid in guids:
        commits.append(
            clock(
                lambda: transaction(
                    session, "remove", lambda: session.remove_object(guid)
                )
            )
        )

    print(
        f"history memory     commit {median(commits[:50]):.4f} -> {median(commits[150:]):.4f} ms, depth {session.history.depth()}, {session.history.bytes} bytes"
    )
    verdict("history commit level", median(commits[150:]) <= 2.0 * median(commits[:50]))


def record_cost(n: int) -> None:
    """Add and remove with a transaction open against the same unrecorded: the record must cost nothing visible."""

    session, group, guids = flat(n)
    plain = [[], []]
    recorded = [[], []]

    for run in range(WARMUP + RUNS):
        x = float(run)
        add = clock(lambda: session.add_point(Point(x, 1.0, 0.0), group))
        remove = clock(lambda: session.remove_object(guids[run]))
        session.begin("record")
        add_recorded = clock(lambda: session.add_point(Point(x, 2.0, 0.0), group))
        remove_recorded = clock(
            lambda: session.remove_object(guids[WARMUP + RUNS + run])
        )
        session.commit()

        if run >= WARMUP:
            plain[0].append(add)
            plain[1].append(remove)
            recorded[0].append(add_recorded)
            recorded[1].append(remove_recorded)

    print(
        f"record cost n={n:<8} add {median(plain[0]):.4f} vs {median(recorded[0]):.4f} ms, remove {median(plain[1]):.4f} vs {median(recorded[1]):.4f} ms"
    )
    cheap = all(median(recorded[k]) - median(plain[k]) < 0.02 * SCALE for k in range(2))
    verdict("record cost", not unthrottled() or cheap)


def layer_move(unrelated_sizes: list[int]) -> None:
    """Moving 10k nodes between two groups with 1k then 100k unrelated objects: the cost must not follow the unrelated count."""

    medians = []

    for unrelated in unrelated_sizes:
        session = Session()
        a = session.add_group("a")
        b = session.add_group("b")
        elsewhere = session.add_group("elsewhere")
        nodes = []
        times = [[] for _ in range(3)]

        for i in range(10_000):
            nodes.append(session.add_point(Point(float(i), 0.0, 0.0), a))

        for i in range(unrelated):
            session.add_point(Point(float(i), 1.0, 0.0), elsewhere)

        for run in range(WARMUP + RUNS):
            target = b if run % 2 == 0 else a

            def move():
                for node in nodes:
                    session.add(node, target)

            lap = [
                clock(lambda: transaction(session, "move", move)),
                clock(session.undo),
                clock(session.redo),
            ]

            while session.purge_step(PURGE_WORK):
                pass

            if run >= WARMUP:
                for k, lapse in enumerate(lap):
                    times[k].append(lapse)

        row = [median(laps) for laps in times]
        print(
            f"layer move unrelated={unrelated:<7} move {row[0]:.2f} undo {row[1]:.2f} redo {row[2]:.2f} ms"
        )
        medians.append(row)

    verdict(
        "layer move budget",
        not unthrottled() or all(lapse < 50.0 * SCALE for lapse in medians[1]),
    )
    verdict(
        "layer move level", all(medians[1][k] < 3.0 * medians[0][k] for k in range(3))
    )


sizes = [LARGEST // 1_000, LARGEST // 10, LARGEST]
bulk = LARGEST // 10
print(
    f"sizes {sizes}, bulk {bulk}, cpu {'unthrottled' if unthrottled() else 'throttled'}"
)
edit_latency(sizes)
bulk_undo(LARGEST, bulk)
no_pauses(LARGEST, bulk)
steady_state(sizes[1])
history_memory()
record_cost(sizes[1])
layer_move([sizes[0], sizes[1]])

"""
description: undo/redo cost of the tombstone kernel, printed, never asserted; the Python port of session_rust/examples/bench.rs.

directory: cd ~/code/code_cpp/session_worktrees/tomb/session/session_py
run: timeout 10m systemd-run --user --scope -p MemoryMax=6G ../uvsession/bin/python examples/bench.py
"""
