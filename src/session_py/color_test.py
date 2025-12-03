from .color import Color
from .mini_test import MINI_TEST, MINI_CHECK, run_all


@MINI_TEST("Color", "constructor")
def test_color_constructor():
    from session_py import Color

    # Constructor
    red = Color(255, 0, 0, 255, "red")

    # Setters
    red[0] = 255
    red[1] = 0
    red[2] = 0
    red[3] = 255

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
    cother = Color(255, 0, 0, 255, "red")

    MINI_CHECK(red.name == "red" and
               red.guid != "" and
               red[0 ] == 255 and
               red[1] == 0 and
               red[2] == 0 and
               red[3] == 255 and
               red.guid)

    MINI_CHECK(r == 255 and g == 0 and b == 0 and a == 255)
    MINI_CHECK(cstr == "255, 0, 0, 255")
    MINI_CHECK(crepr == "Color(red, 255, 0, 0, 255)")
    MINI_CHECK(ccopy == cother)
    MINI_CHECK(ccopy.guid != red.guid)


@MINI_TEST("Color", "conversion")
def test_color_conversion():
    from session_py import Color

    color = Color(255, 128, 64, 255)
    flts = color.to_float_array()
    ints = Color.from_float(flts[0], flts[1], flts[2], flts[3])
    MINI_CHECK([round(flts[0], 3), round(flts[1], 3), round(flts[2], 3), round(flts[3], 3)] == [1.0, 0.502, 0.251, 1.0])
    MINI_CHECK(ints == color)

@MINI_TEST("Color", "presets")
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
    MINI_CHECK(white == Color(255, 255, 255, 255, "white"))
    MINI_CHECK(black == Color(0, 0, 0, 255, "black"))
    MINI_CHECK(grey == Color(128, 128, 128, 255, "grey"))
    MINI_CHECK(red == Color(255, 0, 0, 255, "red"))
    MINI_CHECK(orange == Color(255, 128, 0, 255, "orange"))
    MINI_CHECK(yellow == Color(255, 255, 0, 255, "yellow"))
    MINI_CHECK(lime == Color(128, 255, 0, 255, "lime"))
    MINI_CHECK(green == Color(0, 255, 0, 255, "green"))
    MINI_CHECK(mint == Color(0, 255, 128, 255, "mint"))
    MINI_CHECK(cyan == Color(0, 255, 255, 255, "cyan"))
    MINI_CHECK(azure == Color(0, 128, 255, 255, "azure"))
    MINI_CHECK(blue == Color(0, 0, 255, 255, "blue"))
    MINI_CHECK(violet == Color(128, 0, 255, 255, "violet"))
    MINI_CHECK(magenta == Color(255, 0, 255, 255, "magenta"))
    MINI_CHECK(pink == Color(255, 0, 128, 255, "pink"))
    MINI_CHECK(maroon == Color(128, 0, 0, 255, "maroon"))
    MINI_CHECK(brown == Color(128, 64, 0, 255, "brown"))
    MINI_CHECK(olive == Color(128, 128, 0, 255, "olive"))
    MINI_CHECK(teal == Color(0, 128, 128, 255, "teal"))
    MINI_CHECK(navy == Color(0, 0, 128, 255, "navy"))
    MINI_CHECK(purple == Color(128, 0, 128, 255, "purple"))
    MINI_CHECK(silver == Color(192, 192, 192, 255, "silver"))


@MINI_TEST("Color", "json_roundtrip")
def test_color_json_roundtrip():
    from session_py import Color
    from session_py.encoders import json_dump, json_load
    from pathlib import Path

    color = Color(255, 128, 64, 255)
    color.name = "test_color"
    color.width = 2.0
    color.color = Color(255, 128, 64, 255)

    path = Path(__file__).resolve().parents[2] / "test_color.json"
    json_dump(color, path)
    loaded = json_load(path)

    MINI_CHECK(isinstance(loaded, Color))
    MINI_CHECK(loaded.name == "test_color")
    MINI_CHECK(loaded[0] == 255)
    MINI_CHECK(loaded[1] == 128)
    MINI_CHECK(loaded[2] == 64)
    MINI_CHECK(loaded[3] == 255)

@MINI_TEST("Color", "protobuf_roundtrip")
def test_color_protobuf_roundtrip():
    from session_py import Color
    from pathlib import Path

    color = Color(255, 128, 64, 255)
    color.name = "test_color"
    color.width = 2.0
    color.color = Color(255, 128, 64, 255)

    path = Path(__file__).resolve().parents[2] / "test_color.bin"
    color.protobuf_dump(path)
    loaded = Color.protobuf_load(path)

    MINI_CHECK(isinstance(loaded, Color))
    MINI_CHECK(loaded.name == "test_color")
    MINI_CHECK(loaded[0] == 255)
    MINI_CHECK(loaded[1] == 128)
    MINI_CHECK(loaded[2] == 64)
    MINI_CHECK(loaded[3] == 255)

# def test_color_white():
#     white = Color.white()
#     assert white.name == "white"
#     assert white.r == 255
#     assert white.g == 255
#     assert white.b == 255
#     assert white.a == 255


# def test_color_black():
#     black = Color.black()
#     assert black.name == "black"
#     assert black.r == 0
#     assert black.g == 0
#     assert black.b == 0
#     assert black.a == 255


# def test_color_red():
#     red = Color.red()
#     assert red.name == "red"
#     assert red.r == 255
#     assert red.g == 0
#     assert red.b == 0
#     assert red.a == 255


# def test_color_green():
#     green = Color.green()
#     assert green.name == "green"
#     assert green.r == 0
#     assert green.g == 255
#     assert green.b == 0
#     assert green.a == 255


# def test_color_blue():
#     blue = Color.blue()
#     assert blue.name == "blue"
#     assert blue.r == 0
#     assert blue.g == 0
#     assert blue.b == 255
#     assert blue.a == 255


# def test_color_silver():
#     silver = Color.silver()
#     assert silver.name == "silver"
#     assert silver.r == 192
#     assert silver.g == 192
#     assert silver.b == 192
#     assert silver.a == 255


# def test_color_grey():
#     grey = Color.grey()
#     assert grey.name == "grey"
#     assert grey.r == 128
#     assert grey.g == 128
#     assert grey.b == 128
#     assert grey.a == 255


if __name__ == "__main__":
    run_all(language="python")
