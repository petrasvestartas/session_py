from .mini_test import MINI_TEST
from .mini_test import MINI_CHECK
from .mini_test import run_all
from .tolerance import TOLERANCE


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
    from session_py import History
    from session_py import Session
    from session_py import Point

    session = Session()
    history = session.history

    # An empty transaction is dropped, and nothing is recorded while none is open.
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
    from session_py import Session
    from session_py import Point

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
    from session_py import Session
    from session_py import Point

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
    MINI_CHECK(len(session.objects.points) == 1)


if __name__ == "__main__":
    run_all(language="python")
