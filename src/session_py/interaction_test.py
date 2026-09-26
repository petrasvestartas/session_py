from .mini_test import MINI_TEST
from .mini_test import MINI_CHECK
from .mini_test import run_all


def _named_interaction_class():
    """A test-only subclass: a named interaction with no state of its own."""

    from session_py import Interaction

    class NamedInteraction(Interaction):
        def interaction_type_name(self):
            """Return the registered type name."""
            return "NamedInteraction"

        def interaction_data_dumps(self):
            """Return no state."""
            return b""

    return NamedInteraction


def _named_interaction(data):
    """Build a NamedInteraction from its data."""
    return _named_interaction_class()()


@MINI_TEST("Interaction", "Constructor")
def test_interaction_constructor():
    import copy

    NamedInteraction = _named_interaction_class()
    unnamed = NamedInteraction()
    glue = NamedInteraction("glue")
    duplicate = copy.copy(glue)
    cloned = glue.clone()

    MINI_CHECK(unnamed.name == "")
    MINI_CHECK(glue.name == "glue")
    MINI_CHECK(glue.interaction_type_name() == "NamedInteraction")
    MINI_CHECK(duplicate == glue)
    MINI_CHECK(duplicate.guid == glue.guid)
    MINI_CHECK(cloned == glue)
    MINI_CHECK(cloned.guid == glue.guid)
    MINI_CHECK(unnamed != glue)
    MINI_CHECK(unnamed.guid != glue.guid)
    MINI_CHECK(str(glue) == "NamedInteraction(glue)")
    MINI_CHECK(repr(glue) == "NamedInteraction(" + glue.guid + ", glue)")


@MINI_TEST("Interaction", "Abstract Base")
def test_interaction_abstract_base():
    from session_py import Interaction

    glue = _named_interaction_class()("glue")
    abstract = False

    try:
        Interaction("glue")
    except TypeError:
        abstract = True

    MINI_CHECK(abstract)
    MINI_CHECK(glue.interaction_type_name() == "NamedInteraction")


@MINI_TEST("Interaction", "Json Roundtrip")
def test_interaction_json_roundtrip():
    from pathlib import Path
    from session_py import Interaction

    Interaction.register_type("NamedInteraction", _named_interaction)
    glue = _named_interaction_class()("glue")

    data = glue.__jsondump__()
    loaded_j = Interaction.__jsonload__(data)
    loaded_s = Interaction.file_json_loads(glue.file_json_dumps())

    filename = (
        Path(__file__).resolve().parents[2] / "serialization" / "test_interaction.json"
    )
    glue.file_json_dump(filename)
    loaded = Interaction.file_json_load(filename)

    MINI_CHECK(data["type"] == "Interaction")
    MINI_CHECK(loaded_j == glue)
    MINI_CHECK(loaded_s == glue)
    MINI_CHECK(loaded == glue)
    MINI_CHECK(loaded.guid == glue.guid)


@MINI_TEST("Interaction", "Protobuf Roundtrip")
def test_interaction_protobuf_roundtrip():
    from pathlib import Path
    from session_py import Interaction

    Interaction.register_type("NamedInteraction", _named_interaction)
    glue = _named_interaction_class()("glue")

    proto = glue.to_proto()
    converted = Interaction.from_proto(proto)
    loaded_b = Interaction.pb_loads(glue.pb_dumps())

    filename = (
        Path(__file__).resolve().parents[2] / "serialization" / "test_interaction.bin"
    )
    glue.pb_dump(filename)
    loaded = Interaction.pb_load(filename)

    MINI_CHECK(proto.interaction_type == "NamedInteraction")
    MINI_CHECK(converted == glue)
    MINI_CHECK(loaded_b == glue)
    MINI_CHECK(loaded == glue)
    MINI_CHECK(loaded.guid == glue.guid)


@MINI_TEST("Interaction", "Registry Unknown Type")
def test_interaction_registry_unknown_type():
    from session_py import Interaction
    from session_py import InteractionUnknown

    proto = _named_interaction_class()("mystery").to_proto()
    proto.interaction_type = "NeverRegistered"
    proto.interaction_data = b"whatever this package meant"
    loaded = Interaction.pb_loads(proto.SerializeToString())
    saved = Interaction.pb_loads(loaded.pb_dumps())

    MINI_CHECK(not Interaction.is_registered("NeverRegistered"))
    MINI_CHECK(isinstance(loaded, InteractionUnknown))
    MINI_CHECK(loaded.name == "mystery")
    MINI_CHECK(loaded.interaction_type_name() == "NeverRegistered")
    MINI_CHECK(loaded.interaction_data_dumps() == b"whatever this package meant")
    MINI_CHECK(saved == loaded)


if __name__ == "__main__":
    run_all("python")
