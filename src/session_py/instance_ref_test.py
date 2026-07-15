from .mini_test import MINI_TEST
from .mini_test import MINI_CHECK
from .mini_test import run_all
from .tolerance import TOLERANCE


@MINI_TEST("InstanceRef", "Constructor")
def test_instance_ref_constructor():
    from session_py import InstanceRef
    from session_py import Xform

    # Constructor from a definition guid and a placement transform
    x = Xform.translation(10.0, 20.0, 30.0)
    r = InstanceRef("def-123", x.duplicate())

    # Setter on a copy (keep r pristine for the == check below)
    rset = r.duplicate()
    rset[0] = 2.0
    m0 = rset[0]

    # Minimal and Full String Representation
    rstr = str(r)
    rrepr = repr(r)

    # Copy (duplicate everything but guid)
    rcopy = r.duplicate()
    rother = InstanceRef("def-123", x.duplicate())

    # with_name constructor
    rwn = InstanceRef.with_name("custom", "def-9", Xform.identity())

    MINI_CHECK(r.name == "my_instance_ref")
    MINI_CHECK(r.definition_guid == "def-123")
    MINI_CHECK(len(r.guid) > 0)
    MINI_CHECK(m0 == 2.0)
    MINI_CHECK(r[12] == 10.0 and r[13] == 20.0 and r[14] == 30.0)
    MINI_CHECK("def-123" in rstr)
    MINI_CHECK("InstanceRef" in rrepr and "my_instance_ref" in rrepr)
    MINI_CHECK(rcopy.guid != r.guid)
    MINI_CHECK(r == rother)
    MINI_CHECK(r != rwn)
    MINI_CHECK(rwn.name == "custom" and rwn.definition_guid == "def-9")


@MINI_TEST("InstanceRef", "Transformation")
def test_instance_ref_transformation():
    from session_py import InstanceRef
    from session_py import Xform

    r = InstanceRef("def", Xform.translation(1.0, 0.0, 0.0))
    moved = r.transformed(Xform.translation(5.0, 0.0, 0.0))  # Make a copy
    r.transform(Xform.translation(5.0, 0.0, 0.0))  # compose in place

    # translation(5) * translation(1) => translation(6)
    MINI_CHECK(TOLERANCE.is_close(moved[12], 6.0))
    MINI_CHECK(TOLERANCE.is_close(r[12], 6.0))


@MINI_TEST("InstanceRef", "Json Roundtrip")
def test_instance_ref_json_roundtrip():
    from session_py import InstanceRef
    from session_py import Xform

    r = InstanceRef("def-abc", Xform.translation(1.0, 2.0, 3.0))
    r.name = "test_ref"
    r.flags = 7

    # JSON object
    j = r.__jsondump__()
    loaded_j = InstanceRef.__jsonload__(j)

    MINI_CHECK(loaded_j.name == "test_ref")
    MINI_CHECK(loaded_j.definition_guid == "def-abc")
    MINI_CHECK(loaded_j.flags == 7)
    MINI_CHECK(TOLERANCE.is_close(loaded_j[12], 1.0))

    # String
    s = r.file_json_dumps()
    loaded_s = InstanceRef.file_json_loads(s)
    MINI_CHECK(loaded_s.name == "test_ref")
    MINI_CHECK(loaded_s.definition_guid == "def-abc")

    # File
    fname = "serialization/test_instance_ref.json"
    r.file_json_dump(fname)
    loaded = InstanceRef.file_json_load(fname)
    MINI_CHECK(loaded.name == "test_ref")
    MINI_CHECK(loaded.definition_guid == "def-abc")
    MINI_CHECK(loaded.flags == 7)
    MINI_CHECK(TOLERANCE.is_close(loaded[12], 1.0))
    MINI_CHECK(TOLERANCE.is_close(loaded[13], 2.0))
    MINI_CHECK(TOLERANCE.is_close(loaded[14], 3.0))


@MINI_TEST("InstanceRef", "Protobuf Roundtrip")
def test_instance_ref_protobuf_roundtrip():
    from session_py import InstanceRef
    from session_py import Xform

    r = InstanceRef("def-xyz", Xform.translation(1.0, 2.0, 3.0))
    r.name = "test_ref"
    r.flags = 5

    # Bytes
    b = r.pb_dumps()
    loaded_s = InstanceRef.pb_loads(b)

    MINI_CHECK(loaded_s.name == "test_ref")
    MINI_CHECK(loaded_s.definition_guid == "def-xyz")
    MINI_CHECK(loaded_s.flags == 5)
    MINI_CHECK(loaded_s.guid == r.guid)
    MINI_CHECK(TOLERANCE.is_close(loaded_s[14], 3.0))

    # File
    fname = "serialization/test_instance_ref.bin"
    r.pb_dump(fname)
    loaded = InstanceRef.pb_load(fname)
    MINI_CHECK(loaded.name == "test_ref")
    MINI_CHECK(loaded.definition_guid == "def-xyz")
    MINI_CHECK(loaded.guid == r.guid)
    MINI_CHECK(TOLERANCE.is_close(loaded[12], 1.0))
    MINI_CHECK(TOLERANCE.is_close(loaded[13], 2.0))
    MINI_CHECK(TOLERANCE.is_close(loaded[14], 3.0))


if __name__ == "__main__":
    run_all("python")
