from .mini_test import MINI_TEST
from .mini_test import MINI_CHECK
from .mini_test import run_all
from .tolerance import TOLERANCE


@MINI_TEST("InstanceRef", "Constructor")
def test_instance_ref_constructor():
    from session_py import InstanceRef
    from session_py import Xform

    x = Xform.translation(10.0, 20.0, 30.0)
    inst = InstanceRef("def-123", x)

    instset = inst.duplicate()
    instset[0] = 2.0
    m0 = instset[0]

    istr = str(inst)
    irepr = repr(inst)

    instcopy = inst.duplicate()
    instother = InstanceRef("def-123", x)
    named = InstanceRef.with_name("custom", "def-9", Xform.identity())

    MINI_CHECK(inst.name == "my_instance_ref")
    MINI_CHECK(inst.definition_guid == "def-123")
    MINI_CHECK(len(inst.guid) > 0)
    MINI_CHECK(m0 == 2.0)
    MINI_CHECK(inst[12] == 10.0 and inst[13] == 20.0 and inst[14] == 30.0)
    MINI_CHECK("def-123" in istr)
    MINI_CHECK("InstanceRef" in irepr)
    MINI_CHECK("my_instance_ref" in irepr)
    MINI_CHECK(instcopy.guid != inst.guid)
    MINI_CHECK(inst == instother)
    MINI_CHECK(inst != named)
    MINI_CHECK(named.name == "custom" and named.definition_guid == "def-9")


@MINI_TEST("InstanceRef", "Transformation")
def test_instance_ref_transformation():
    from session_py import InstanceRef
    from session_py import Xform

    inst = InstanceRef("def", Xform.translation(1.0, 0.0, 0.0))
    moved = inst.transformed(Xform.translation(5.0, 0.0, 0.0))
    inst.transform(Xform.translation(5.0, 0.0, 0.0))

    MINI_CHECK(TOLERANCE.is_close(moved[12], 6.0))
    MINI_CHECK(TOLERANCE.is_close(inst[12], 6.0))


@MINI_TEST("InstanceRef", "Json Roundtrip")
def test_instance_ref_json_roundtrip():
    from session_py import InstanceRef
    from session_py import Xform

    inst = InstanceRef("def-abc", Xform.translation(1.0, 2.0, 3.0))
    inst.name = "test_ref"
    inst.flags = 7

    j = inst.__jsondump__()
    loaded_j = InstanceRef.__jsonload__(j)

    MINI_CHECK(loaded_j.name == "test_ref")
    MINI_CHECK(loaded_j.definition_guid == "def-abc")
    MINI_CHECK(loaded_j.flags == 7)
    MINI_CHECK(TOLERANCE.is_close(loaded_j[12], 1.0))

    s = inst.file_json_dumps()
    loaded_s = InstanceRef.file_json_loads(s)

    MINI_CHECK(loaded_s.name == "test_ref")
    MINI_CHECK(loaded_s.definition_guid == "def-abc")

    filename = "serialization/test_instance_ref.json"
    inst.file_json_dump(filename)
    loaded = InstanceRef.file_json_load(filename)

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

    inst = InstanceRef("def-xyz", Xform.translation(1.0, 2.0, 3.0))
    inst.name = "test_ref"
    inst.flags = 5

    guid = inst.guid
    b = inst.pb_dumps()
    loaded_b = InstanceRef.pb_loads(b)

    MINI_CHECK(loaded_b.name == "test_ref")
    MINI_CHECK(loaded_b.definition_guid == "def-xyz")
    MINI_CHECK(loaded_b.flags == 5)
    MINI_CHECK(loaded_b.guid == guid)
    MINI_CHECK(TOLERANCE.is_close(loaded_b[14], 3.0))

    filename = "serialization/test_instance_ref.bin"
    inst.pb_dump(filename)
    loaded = InstanceRef.pb_load(filename)

    MINI_CHECK(loaded.name == "test_ref")
    MINI_CHECK(loaded.definition_guid == "def-xyz")
    MINI_CHECK(loaded.guid == guid)
    MINI_CHECK(TOLERANCE.is_close(loaded[12], 1.0))
    MINI_CHECK(TOLERANCE.is_close(loaded[13], 2.0))
    MINI_CHECK(TOLERANCE.is_close(loaded[14], 3.0))


if __name__ == "__main__":
    run_all("python")
