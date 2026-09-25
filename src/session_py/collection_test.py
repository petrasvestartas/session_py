from .mini_test import MINI_TEST
from .mini_test import MINI_CHECK
from .mini_test import run_all


@MINI_TEST("Collection", "Constructor")
def test_collection_constructor():
    from session_py import Collection
    from session_py import Point

    a = Point(0.0, 0.0, 0.0)
    b = Point(1.0, 0.0, 0.0)
    c = Point(2.0, 0.0, 0.0)
    points = Collection()
    points.append(a)
    points.append(b)
    points.append(c)
    iterated = list(points)

    MINI_CHECK(len(points) == 3)
    MINI_CHECK(points[0] is a and points[1] is b)
    MINI_CHECK(points[2] is c)
    MINI_CHECK(iterated[0] is a and iterated[2] is c)
    MINI_CHECK(points.get_slot(b.guid) == 1)
    MINI_CHECK(points.number_of_slots() == 3)
    MINI_CHECK(points.number_of_dead() == 0)
    MINI_CHECK(str(points) == "Collection(3 live, 0 dead)")


@MINI_TEST("Collection", "Set Dead")
def test_collection_set_dead():
    from session_py import Collection
    from session_py import Point

    a = Point(0.0, 0.0, 0.0)
    b = Point(1.0, 0.0, 0.0)
    c = Point(2.0, 0.0, 0.0)
    points = Collection([a, b, c])
    points.set_dead(1, True)
    killed = list(points)
    killed_len = len(points)
    killed_slot = points.get_slot(b.guid)
    killed_dead = points.number_of_dead()
    killed_slots = points.number_of_slots()
    kept = points.get_item(1) is b and points.is_dead(1)
    second = points[1] is c
    points.set_dead(1, False)

    MINI_CHECK(killed_len == 2 and second)
    MINI_CHECK(killed[0] is a and killed[1] is c)
    MINI_CHECK(killed_slot is None and kept)
    MINI_CHECK(killed_dead == 1 and killed_slots == 3)
    MINI_CHECK(len(points) == 3)
    MINI_CHECK(points[1] is b and points[2] is c)
    MINI_CHECK(points.get_slot(b.guid) == 1)


@MINI_TEST("Collection", "Index Skips Dead")
def test_collection_index_skips_dead():
    from session_py import Collection
    from session_py import Point

    e = []

    for i in range(6):
        e.append(Point(float(i), 0.0, 0.0))

    points = Collection(e[0:5])
    points.set_dead(0, True)
    points.set_dead(3, True)
    missing = _raises_index_error(points, 3)
    points.append(e[5])

    MINI_CHECK(points[0] is e[1] and points[1] is e[2])
    MINI_CHECK(points[2] is e[4] and points[-2] is e[4])
    MINI_CHECK(points[0] is e[1] and points[0:2] == [e[1], e[2]])
    MINI_CHECK(missing)
    MINI_CHECK(points[3] is e[5])
    MINI_CHECK(points[-1] is e[5])


@MINI_TEST("Collection", "Compact")
def test_collection_compact():
    from session_py import Collection
    from session_py import Point
    from session_py.history import Tomb

    e = []

    for i in range(5):
        e.append(Point(float(i), 0.0, 0.0))

    points = Collection(e)
    tomb = Tomb("points", False, 3, None)
    points.set_dead(1, True)
    points.set_dead(3, True)
    points.set_tomb(3, tomb)
    points.compact()
    pinned_slots = points.number_of_slots()
    pinned_dead = points.number_of_dead()
    order = list(points)
    slots = [
        points.get_slot(e[0].guid),
        points.get_slot(e[2].guid),
        points.get_slot(e[4].guid),
    ]
    moved = tomb.slot
    del tomb
    points.compact()

    MINI_CHECK(pinned_slots == 4 and pinned_dead == 1 and moved == 2)
    MINI_CHECK(order[0] is e[0] and order[1] is e[2])
    MINI_CHECK(order[2] is e[4])
    MINI_CHECK(slots == [0, 1, 3])
    MINI_CHECK(points.number_of_slots() == 3 and points.number_of_dead() == 0)
    MINI_CHECK(points.get_slot(e[4].guid) == 2)


@MINI_TEST("Collection", "Compact Step")
def test_collection_compact_step():
    from session_py import Collection
    from session_py import Point
    from session_py.history import Tomb

    points = Collection()
    model = []
    tombs = []

    for i in range(1000):
        point = Point(float(i), 0.0, 0.0)
        points.append(point)
        model.append([point, i % 3 != 0])

    for i in range(0, 1000, 3):
        points.set_dead(i, True)

        if len(tombs) < 10:
            tombs.append(Tomb("points", False, i, None))
            points.set_tomb(i, tombs[len(tombs) - 1])

    bounded = True
    exact = True
    revived = 0

    while True:
        bounded &= points.compact_step(10) <= 10
        expected = []

        for m in model:
            if m[1]:
                expected.append(m[0])

        exact &= len(points) == len(expected)
        exact &= all(p is q for p, q in zip(points, expected))

        for p in points:
            s = points.get_slot(p.guid)
            exact &= s is not None and points.get_item(s) is p

        if not points.is_compacting():
            break

        point = Point(-1.0, 0.0, 0.0)
        points.append(point)
        model.append([point, True])

        if revived < 5:
            points.set_dead(tombs[revived].slot, False)
            model[revived * 3][1] = True
            revived += 1

    settled = points.number_of_slots() == len(points) + 5
    points.compact_step(10)
    points.set_dead(1, True)

    while points.is_compacting():
        points.compact_step(10)

    waiting = points.is_dead(1) and points.number_of_dead() == 6
    points.compact()

    MINI_CHECK(bounded and exact and settled)
    MINI_CHECK(waiting)
    MINI_CHECK(points.number_of_dead() == 5)
    MINI_CHECK(points.number_of_slots() == len(points) + 5)


@MINI_TEST("Collection", "Json Roundtrip")
def test_collection_json_roundtrip():
    from session_py import Collection
    from session_py import Point
    from session_py.file_encoders import file_json_dumps
    import json

    points = Collection()
    points.append(Point(0.0, 0.0, 0.0))
    points.append(Point(1.0, 0.0, 0.0))
    points.append(Point(2.0, 0.0, 0.0))
    points.set_dead(1, True)
    data = file_json_dumps(points)
    loaded = Collection.__jsonload__(json.loads(data))

    MINI_CHECK(data == file_json_dumps(list(points)))
    MINI_CHECK(len(loaded) == 2 and loaded.number_of_dead() == 0)
    MINI_CHECK(loaded[0].guid == points[0].guid)
    MINI_CHECK(loaded[1].guid == points[1].guid)


def _raises_index_error(points, i: int) -> bool:
    """Whether reading live position i raises IndexError."""

    try:
        points[i]
    except IndexError:
        return True

    return False


if __name__ == "__main__":
    run_all(language="python")
