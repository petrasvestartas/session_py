from .mini_test import MINI_TEST
from .mini_test import MINI_CHECK
from .mini_test import run_all
from .tolerance import TOLERANCE


def grid_mesh(n):
    """A mesh of n by n vertices in quads, one guid of its own."""

    from session_py import Mesh
    from session_py import Point

    vertices = []
    faces = []

    for i in range(n):
        for j in range(n):
            vertices.append(Point(float(i), float(j), 0.0))

    for i in range(n - 1):
        for j in range(n - 1):
            at = i * n + j
            faces.append([at, at + n, at + n + 1, at + 1])

    return Mesh.from_vertices_and_faces(vertices, faces)


@MINI_TEST("History", "Constructor")
def test_history_constructor():
    from session_py import History

    history = History()
    hstr = str(history)
    hrepr = repr(history)

    MINI_CHECK(not history.can_undo())
    MINI_CHECK(not history.can_redo())
    MINI_CHECK(history.depth() == 0)
    MINI_CHECK(hstr == "History(0 undo, 0 redo)")
    MINI_CHECK(hrepr == "History(0 undo, 0 redo)")


@MINI_TEST("History", "Begin Commit")
def test_history_begin_commit():
    from session_py import Point
    from session_py import Session

    session = Session()
    history = session.history

    history.begin("empty")
    history.commit()
    session.add_point(Point(0.0, 0.0, 0.0))

    history.begin("add")
    session.add_point(Point(1.0, 0.0, 0.0))
    history.commit()

    MINI_CHECK(history.depth() == 1)
    MINI_CHECK(history.can_undo())
    MINI_CHECK(len(history.undo_stack[0].ops) == 1)
    MINI_CHECK(history.undo_stack[0].label == "add")
    MINI_CHECK(history.undo_stack[0].ops[0].kind == "add")


@MINI_TEST("History", "Undo Redo")
def test_history_undo_redo():
    from session_py import Point
    from session_py import Session

    session = Session()
    point = Point(1.0, 2.0, 3.0)
    guid = point.guid

    session.history.begin("add")
    session.add_point(point)
    session.history.commit()

    undone = session.history.undo(session)
    absent = guid not in session.lookup
    redone = session.history.redo(session)

    MINI_CHECK(undone)
    MINI_CHECK(absent)
    MINI_CHECK(redone)
    MINI_CHECK(guid in session.lookup)
    MINI_CHECK(TOLERANCE.is_close(session.lookup[guid][2], 3.0))
    MINI_CHECK(not session.history.can_redo())
    MINI_CHECK(not session.history.redo(session))


@MINI_TEST("History", "Clear")
def test_history_clear():
    from session_py import Point
    from session_py import Session

    session = Session()

    session.history.begin("a")
    session.add_point(Point(0.0, 0.0, 0.0))
    session.history.commit()

    session.history.begin("b")
    session.add_point(Point(1.0, 0.0, 0.0))
    session.history.commit()

    session.history.undo(session)
    session.history.clear()

    MINI_CHECK(not session.history.can_undo())
    MINI_CHECK(not session.history.can_redo())
    MINI_CHECK(session.history.depth() == 0)
    MINI_CHECK(session.history.bytes == 0)
    MINI_CHECK(session.history.dropped == 2)
    MINI_CHECK(len(session.objects.points) == 1)
    MINI_CHECK(session.objects.points.number_of_dead() == 1)


@MINI_TEST("History", "Undo Definition")
def test_history_undo_definition():
    from session_py import Point
    from session_py import Session

    session = Session()
    point = Point(1.0, 2.0, 3.0)
    guid = point.guid

    session.begin("define")
    session.add_definition(point)
    session.commit()

    session.begin("replace")
    session.replace_definition(guid, Point(9.0, 9.0, 9.0))
    session.commit()

    session.begin("remove")
    session.remove_definition(guid)
    session.commit()

    defined = repr(session.history.undo_stack[0].ops[0])
    swapped = session.history.undo_stack[1].ops[0].kind
    dropped = repr(session.history.undo_stack[2].ops[0])
    removed = guid not in session.definition_lookup
    session.undo()
    replaced = session.definition_lookup[guid][0]
    session.undo()
    restored = session.definition_lookup[guid][0]
    session.undo()
    undefined = (
        len(session.definition_lookup) == 0 and len(session.definitions.points) == 0
    )
    redone = session.redo()

    MINI_CHECK(defined == f"add({guid}, definitions)")
    MINI_CHECK(swapped == "replace")
    MINI_CHECK(dropped == f"remove({guid}, definitions)")
    MINI_CHECK(removed)
    MINI_CHECK(TOLERANCE.is_close(replaced, 9.0))
    MINI_CHECK(TOLERANCE.is_close(restored, 1.0))
    MINI_CHECK(undefined)
    MINI_CHECK(redone)
    MINI_CHECK(len(session.definitions.points) == 1)
    MINI_CHECK(session.definitions.points.number_of_slots() == 1)
    MINI_CHECK(session.definitions.points[0].guid == guid)


@MINI_TEST("History", "Budget")
def test_history_budget():
    from session_py import Session

    session = Session()
    session.history.budget = 1 << 20
    guids = []

    for _ in range(20):
        mesh = grid_mesh(100)
        guids.append(mesh.guid)
        session.add_mesh(mesh)

    for guid in guids:
        session.begin("remove")
        session.remove_object(guid)
        session.commit()

    newest = session.history.undo_stack[session.history.depth() - 1].bytes
    pinned = 0

    for transaction in session.history.undo_stack + session.history.redo_stack:
        pinned += transaction.bytes

    MINI_CHECK(session.history.depth() < 20)
    MINI_CHECK(session.history.bytes <= session.history.budget + newest)
    MINI_CHECK(session.history.dropped > 0)
    MINI_CHECK(session.history.bytes == pinned)
    MINI_CHECK(session.undo())
    MINI_CHECK(guids[19] in session.lookup)


@MINI_TEST("History", "Weight")
def test_history_weight():
    from session_py import Point
    from session_py import Session
    from session_py.history import RECORD
    from session_py.history import weight

    point = Point(0.0, 0.0, 0.0)
    small = grid_mesh(32)
    large = grid_mesh(100)
    session = Session()
    guid = large.guid
    a = Point(0.0, 0.0, 0.0)
    b = Point(1.0, 0.0, 0.0)
    a_guid = a.guid
    b_guid = b.guid
    session.add_mesh(large)
    session.add_point(a)
    session.add_point(b)
    session.add_edge(a_guid, guid, "touch")
    session.add_edge(b_guid, guid, "touch")

    session.begin("remove")
    session.remove_object(guid)
    removed = session.history.current.bytes
    session.add_point(Point(2.0, 0.0, 0.0))
    added = session.history.current.bytes
    session.commit()

    MINI_CHECK(weight(point) < weight(small))
    MINI_CHECK(weight(small) < weight(large))
    MINI_CHECK(removed == RECORD + weight(large) + 128 * 2)
    MINI_CHECK(added == removed + RECORD)


@MINI_TEST("History", "Abort")
def test_history_abort():
    from session_py import Point
    from session_py import Session
    from session_py import Xform

    session = Session()
    group = session.add_group("g")
    b = Point(1.0, 0.0, 0.0)
    b_guid = b.guid
    session.add_point(Point(0.0, 0.0, 0.0), group)
    b_node = session.add_point(b, group)
    session.add_point(Point(2.0, 0.0, 0.0), group)
    session.begin("kept")
    session.set_xform(b_guid, Xform.translation(0.0, 1.0, 0.0))
    session.commit()
    session.undo()
    a = Point(5.0, 0.0, 0.0)
    a_guid = a.guid

    session.begin("aborted")
    session.add_point(a, group)
    session.remove_object(b_guid)
    aborted = session.abort()

    MINI_CHECK(aborted)
    MINI_CHECK(a_guid not in session.lookup)
    MINI_CHECK(a_guid not in session.order())
    MINI_CHECK(session.get_node(a_guid) is None)
    MINI_CHECK(b_guid in session.lookup)
    MINI_CHECK(session.objects.points.get_slot(b_guid) == 1)
    MINI_CHECK(group.children[1] is b_node)
    MINI_CHECK(b_node.at() == 1)
    MINI_CHECK(session.history.depth() == 0)
    MINI_CHECK(session.history.can_redo())
    MINI_CHECK(len(session.history.redo_stack) == 1)
    MINI_CHECK(not session.abort())


if __name__ == "__main__":
    run_all(language="python")
