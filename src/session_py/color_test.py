from .color import Color
from .mini_test import MINI_TEST
from .mini_test import MINI_CHECK
from .mini_test import run_all
from .tolerance import TOLERANCE


@MINI_TEST("Color", "Constructor")
def test_color_constructor():
    from session_py import Color

    # Constructor
    red = Color(1.0, 0.0, 0.0, 1.0, "red")

    # Setters
    red[0] = 1.0
    red[1] = 0.0
    red[2] = 0.0
    red[3] = 1.0

    # Getters
    r = red[0]
    g = red[1]
    b = red[2]
    a = red[3]

    # Minimal and Full String Representation
    cstr = str(red)
    crepr = repr(red)

    # Copy (duplicates everything except guid)
    ccopy = red.duplicate()
    cother = Color(1.0, 0.0, 0.0, 1.0, "red")

    MINI_CHECK(red.name == "red")
    MINI_CHECK(red.guid != "")
    MINI_CHECK(red[0] == 1.0)
    MINI_CHECK(red[1] == 0.0)
    MINI_CHECK(red[2] == 0.0)
    MINI_CHECK(red[3] == 1.0)
    MINI_CHECK(red.guid)

    MINI_CHECK(r == 1.0 and g == 0.0 and b == 0.0 and a == 1.0)
    MINI_CHECK(cstr == "1.0, 0.0, 0.0, 1.0")
    MINI_CHECK(crepr == "Color(red, 1.0, 0.0, 0.0, 1.0)")
    MINI_CHECK(ccopy == cother)
    MINI_CHECK(ccopy.guid != red.guid)


@MINI_TEST("Color", "Json Roundtrip")
def test_color_json_roundtrip():
    from session_py import Color
    from pathlib import Path

    c = Color(1.0, 0.5, 0.25, 1.0, "test_color")

    #   __jsondump__()  │ dict         │ to JSON object (internal use)
    #   __jsonload__(d) │ dict         │ from JSON object (internal use)
    #   file_json_dumps()    │ str          │ to JSON string
    #   file_json_loads(s)   │ str          │ from JSON string
    #   file_json_dump(path) │ file         │ write to file
    #   file_json_load(path) │ file         │ read from file

    # file_json_dump(fname) / file_json_load(fname) - file-based serialization
    fname = Path(__file__).resolve().parents[2] / "serialization" / "test_color.json"
    c.file_json_dump(fname)
    loaded = Color.file_json_load(fname)

    MINI_CHECK(loaded.name == "test_color")
    MINI_CHECK(loaded[0] == 1.0)
    MINI_CHECK(loaded[1] == 0.5)
    MINI_CHECK(loaded[2] == 0.25)
    MINI_CHECK(loaded[3] == 1.0)


@MINI_TEST("Color", "Protobuf Roundtrip")
def test_color_protobuf_roundtrip():
    from session_py import Color
    from pathlib import Path

    color = Color(1.0, 0.5, 0.25, 1.0, "test_color")

    #   pb_dumps()      │ bytes        │ to protobuf bytes
    #   pb_loads(b)     │ bytes        │ from protobuf bytes
    #   pb_dump(path)   │ file         │ write to file
    #   pb_load(path)   │ file         │ read from file

    path = Path(__file__).resolve().parents[2] / "serialization" / "test_color.bin"
    color.pb_dump(path)
    loaded = Color.pb_load(path)

    MINI_CHECK(loaded.name == "test_color")
    MINI_CHECK(loaded[0] == 1.0)
    MINI_CHECK(loaded[1] == 0.5)
    MINI_CHECK(loaded[2] == 0.25)
    MINI_CHECK(loaded[3] == 1.0)


@MINI_TEST("Color", "Conversion")
def test_color_conversion():
    from session_py import Color

    color = Color(1.0, 0.5, 0.25, 1.0)
    flts = color.to_unified_array()
    ints = Color.from_unified_array(flts)

    MINI_CHECK(TOLERANCE.is_close(flts[0], 1.0))
    MINI_CHECK(TOLERANCE.is_close(flts[1], 0.5))
    MINI_CHECK(TOLERANCE.is_close(flts[2], 0.25))
    MINI_CHECK(TOLERANCE.is_close(flts[3], 1.0))
    MINI_CHECK(ints == color)

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


if __name__ == "__main__":
    run_all(language="python")
