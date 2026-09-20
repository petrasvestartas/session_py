from .mini_test import MINI_TEST
from .mini_test import MINI_CHECK
from .mini_test import run_all
from .tolerance import TOLERANCE


@MINI_TEST("Color", "Constructor")
def test_color_constructor():
    from session_py import Color

    cdefault = Color()
    clamped = Color(-1.0, 2.0, 0.5, 3.0)
    c = Color(1.0, 0.0, 0.0, 1.0, "red")
    fresh = not c.has_guid()

    c[0] = 1.0
    c[1] = 0.0
    c[2] = 0.0
    c[3] = 1.0

    r = c[0]
    g = c[1]
    b = c[2]
    a = c[3]

    cstr = str(c)
    crepr = repr(c)

    ccopy = c.duplicate()
    cother = Color(1.0, 0.0, 0.0, 1.0, "red")

    MINI_CHECK(cdefault == Color(0.94, 0.94, 0.94, 1.0))
    MINI_CHECK(clamped == Color(0.0, 1.0, 0.5, 1.0))
    MINI_CHECK(fresh)
    MINI_CHECK(c.name == "red")
    MINI_CHECK(c.guid != "")
    MINI_CHECK(c[0] == 1.0 and c[1] == 0.0 and c[2] == 0.0 and c[3] == 1.0)
    MINI_CHECK(r == 1.0 and g == 0.0 and b == 0.0 and a == 1.0)
    MINI_CHECK(cstr == "1.0, 0.0, 0.0, 1.0")
    MINI_CHECK(crepr == "Color(red, 1.0, 0.0, 0.0, 1.0)")
    MINI_CHECK(ccopy == cother)
    MINI_CHECK(c != Color.blue())
    MINI_CHECK(ccopy.guid != c.guid)


@MINI_TEST("Color", "Json Roundtrip")
def test_color_json_roundtrip():
    from pathlib import Path
    from session_py import Color

    c = Color(1.0, 0.5, 0.25, 1.0, "test_color")

    guid = c.guid
    filename = Path(__file__).resolve().parents[2] / "serialization" / "test_color.json"
    c.file_json_dump(filename)
    loaded = Color.file_json_load(filename)
    parsed = Color.file_json_loads(c.file_json_dumps())

    MINI_CHECK(loaded.name == "test_color")
    MINI_CHECK(loaded[0] == 1.0)
    MINI_CHECK(loaded[1] == 0.5)
    MINI_CHECK(loaded[2] == 0.25)
    MINI_CHECK(loaded[3] == 1.0)
    MINI_CHECK(parsed == c)
    MINI_CHECK(loaded.guid == guid)
    MINI_CHECK(parsed.guid == guid)


@MINI_TEST("Color", "Protobuf Roundtrip")
def test_color_protobuf_roundtrip():
    from pathlib import Path
    from session_py import Color

    fresh = Color()
    fresh_proto = fresh.to_proto()
    c = Color(1.0, 0.5, 0.25, 1.0, "test_color")

    guid = c.guid
    filename = Path(__file__).resolve().parents[2] / "serialization" / "test_color.bin"
    c.pb_dump(filename)
    loaded = Color.pb_load(filename)
    parsed = Color.pb_loads(c.pb_dumps())
    converted = Color.from_proto(c.to_proto())

    MINI_CHECK(not fresh.has_guid())
    MINI_CHECK(fresh_proto.guid == "")
    MINI_CHECK(loaded.name == "test_color")
    MINI_CHECK(loaded[0] == 1.0)
    MINI_CHECK(loaded[1] == 0.5)
    MINI_CHECK(loaded[2] == 0.25)
    MINI_CHECK(loaded[3] == 1.0)
    MINI_CHECK(parsed == c)
    MINI_CHECK(loaded.guid == guid)
    MINI_CHECK(parsed.guid == guid)
    MINI_CHECK(converted == c)
    MINI_CHECK(converted.guid == guid)


@MINI_TEST("Color", "Conversion")
def test_color_conversion():
    from session_py import Color

    c = Color(1.0, 0.5, 0.25, 1.0)
    flts = c.to_unified_array()
    back = Color.from_unified_array(flts)

    MINI_CHECK(TOLERANCE.is_close(flts[0], 1.0))
    MINI_CHECK(TOLERANCE.is_close(flts[1], 0.5))
    MINI_CHECK(TOLERANCE.is_close(flts[2], 0.25))
    MINI_CHECK(TOLERANCE.is_close(flts[3], 1.0))
    MINI_CHECK(back == c)


@MINI_TEST("Color", "Presets")
def test_color_presets():
    from session_py import Color

    white = Color.white()
    black = Color.black()
    grey = Color.grey()
    red = Color.red()
    orange = Color.orange()
    yellow = Color.yellow()
    lime = Color.lime()
    green = Color.green()
    mint = Color.mint()
    cyan = Color.cyan()
    azure = Color.azure()
    blue = Color.blue()
    violet = Color.violet()
    magenta = Color.magenta()
    pink = Color.pink()
    maroon = Color.maroon()
    brown = Color.brown()
    olive = Color.olive()
    teal = Color.teal()
    navy = Color.navy()
    purple = Color.purple()
    silver = Color.silver()
    lightgrey = Color.lightgrey()
    palette = Color.palette()

    MINI_CHECK(white == Color(1.0, 1.0, 1.0, 1.0, "white"))
    MINI_CHECK(black == Color(0.0, 0.0, 0.0, 1.0, "black"))
    MINI_CHECK(grey == Color(0.5, 0.5, 0.5, 1.0, "grey"))
    MINI_CHECK(red == Color(1.0, 0.0, 0.0, 1.0, "red"))
    MINI_CHECK(orange == Color(1.0, 0.5, 0.0, 1.0, "orange"))
    MINI_CHECK(yellow == Color(1.0, 1.0, 0.0, 1.0, "yellow"))
    MINI_CHECK(lime == Color(0.5, 1.0, 0.0, 1.0, "lime"))
    MINI_CHECK(green == Color(0.0, 1.0, 0.0, 1.0, "green"))
    MINI_CHECK(mint == Color(0.0, 1.0, 0.5, 1.0, "mint"))
    MINI_CHECK(cyan == Color(0.0, 1.0, 1.0, 1.0, "cyan"))
    MINI_CHECK(azure == Color(0.0, 0.5, 1.0, 1.0, "azure"))
    MINI_CHECK(blue == Color(0.0, 0.0, 1.0, 1.0, "blue"))
    MINI_CHECK(violet == Color(0.5, 0.0, 1.0, 1.0, "violet"))
    MINI_CHECK(magenta == Color(1.0, 0.0, 1.0, 1.0, "magenta"))
    MINI_CHECK(pink == Color(1.0, 0.0, 0.5, 1.0, "pink"))
    MINI_CHECK(maroon == Color(0.5, 0.0, 0.0, 1.0, "maroon"))
    MINI_CHECK(brown == Color(0.5, 0.25, 0.0, 1.0, "brown"))
    MINI_CHECK(olive == Color(0.5, 0.5, 0.0, 1.0, "olive"))
    MINI_CHECK(teal == Color(0.0, 0.5, 0.5, 1.0, "teal"))
    MINI_CHECK(navy == Color(0.0, 0.0, 0.5, 1.0, "navy"))
    MINI_CHECK(purple == Color(0.5, 0.0, 0.5, 1.0, "purple"))
    MINI_CHECK(silver == Color(0.75, 0.75, 0.75, 1.0, "silver"))
    MINI_CHECK(lightgrey == Color(0.94, 0.94, 0.94, 1.0, "lightgrey"))
    MINI_CHECK(
        palette
        == [
            red,
            orange,
            yellow,
            lime,
            green,
            mint,
            cyan,
            azure,
            blue,
            violet,
            magenta,
            pink,
        ]
    )


@MINI_TEST("Color", "Serialization Errors")
def test_color_serialization_errors():
    from google.protobuf.message import DecodeError
    from session_py import Color

    color = Color()
    malformed_json = False
    malformed_pb = False
    json_write_failed = False
    pb_write_failed = False

    try:
        Color.file_json_loads("{}")
    except KeyError:
        malformed_json = True

    try:
        Color.pb_loads(b"\xff")
    except DecodeError:
        malformed_pb = True

    try:
        color.file_json_dump("")
    except OSError:
        json_write_failed = True

    try:
        color.pb_dump("")
    except OSError:
        pb_write_failed = True

    MINI_CHECK(malformed_json)
    MINI_CHECK(malformed_pb)
    MINI_CHECK(json_write_failed)
    MINI_CHECK(pb_write_failed)


if __name__ == "__main__":
    run_all(language="python")
