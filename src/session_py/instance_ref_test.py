from .mini_test import MINI_TEST
from .mini_test import MINI_CHECK
from .mini_test import run_all
from .tolerance import TOLERANCE


@MINI_TEST("InstanceRef", "Constructor")
def test_instance_ref_constructor():
    from session_py import ElementFeature
    from session_py import InstanceRef
    from session_py import Point
    from session_py import Polyline
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

    featured = InstanceRef("def-123", x)
    featured.features.append(
        ElementFeature("contact", 0, [Polyline([Point(0, 0, 0), Point(1, 0, 0)])])
    )
    featuredcopy = featured.duplicate()

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
    MINI_CHECK(len(inst.features) == 0)
    MINI_CHECK(inst != featured)
    MINI_CHECK(featuredcopy == featured and featuredcopy.guid != featured.guid)
    MINI_CHECK(
        InstanceRef.FLAG_HIDDEN == 1
        and InstanceRef.FLAG_LOCKED == 2
        and InstanceRef.FLAG_COLOR == 4
    )


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
    from session_py import ElementFeature
    from session_py import InstanceRef
    from session_py import Point
    from session_py import Polyline
    from session_py import Xform

    inst = InstanceRef("def-abc", Xform.translation(1.0, 2.0, 3.0))
    inst.name = "test_ref"
    inst.flags = 7
    inst.features.append(
        ElementFeature("contact", 0, [Polyline([Point(0, 0, 0), Point(1, 0, 0)])])
    )
    feature = inst.features[0].guid

    j = inst.__jsondump__()
    loaded_j = InstanceRef.__jsonload__(j)
    bare = inst.__jsondump__()
    bare.pop("features")
    bare.pop("flags")
    loaded_bare = InstanceRef.__jsonload__(bare)

    MINI_CHECK(loaded_j.name == "test_ref")
    MINI_CHECK(loaded_j.definition_guid == "def-abc")
    MINI_CHECK(loaded_j.flags == 7)
    MINI_CHECK(len(loaded_j.features) == 1)
    MINI_CHECK(loaded_j.features[0].guid == feature)
    MINI_CHECK(loaded_j == inst)
    MINI_CHECK(loaded_bare.flags == 0 and len(loaded_bare.features) == 0)
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
    from session_py import Color
    from session_py import ElementFeature
    from session_py import InstanceRef
    from session_py import Point
    from session_py import Polyline
    from session_py import Xform

    fresh = InstanceRef()
    fresh_proto = fresh.to_proto()
    inst = InstanceRef("def-xyz", Xform.translation(1.0, 2.0, 3.0))
    inst.name = "test_ref"
    inst.flags = 5
    inst.features.append(
        ElementFeature("contact", 0, [Polyline([Point(0, 0, 0), Point(1, 0, 0)])])
    )
    feature = inst.features[0].guid
    plain = InstanceRef("def-xyz", Xform.identity())
    plain.color = Color.red()
    plain_proto = plain.to_proto()
    plain_loaded = InstanceRef.from_proto(plain_proto)

    guid = inst.guid
    b = inst.pb_dumps()
    loaded_b = InstanceRef.pb_loads(b)
    converted = InstanceRef.from_proto(inst.to_proto())

    MINI_CHECK(not fresh.has_guid())
    MINI_CHECK(fresh_proto.guid == "")
    MINI_CHECK(not plain_proto.HasField("xform"))
    MINI_CHECK(not plain_proto.HasField("color"))
    MINI_CHECK(plain_loaded.xform == Xform.identity())
    MINI_CHECK(plain_loaded.color == Color.white())
    MINI_CHECK(len(loaded_b.features) == 1)
    MINI_CHECK(loaded_b.features[0].guid == feature)
    MINI_CHECK(loaded_b.name == "test_ref")
    MINI_CHECK(loaded_b.definition_guid == "def-xyz")
    MINI_CHECK(loaded_b.flags == 5)
    MINI_CHECK(loaded_b.guid == guid)
    MINI_CHECK(TOLERANCE.is_close(loaded_b[14], 3.0))
    MINI_CHECK(converted == inst)
    MINI_CHECK(converted.guid == guid)

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
