from .mini_test import MINI_TEST
from .mini_test import MINI_CHECK
from .mini_test import run_all
from session_py.session_config import SESSION_CONFIG, SessionConfig


@MINI_TEST("SessionConfig", "Runtime Modification")
def test_session_config_runtime_modification():
    # SessionConfig holds global rendering/export flags.
    # explode_mesh_faces: when true, each mesh face is a separate mesh (for coloring)
    # scale_factor: unit conversion multiplier applied during export (1.0 = meters)
    # Modify config at runtime, then reset() restores all defaults.

    MINI_CHECK(not SESSION_CONFIG.explode_mesh_faces)
    SESSION_CONFIG.explode_mesh_faces = True
    MINI_CHECK(SESSION_CONFIG.explode_mesh_faces)
    MINI_CHECK(SESSION_CONFIG.scale_factor == 1.0)
    SESSION_CONFIG.scale_factor = 0.001
    MINI_CHECK(SESSION_CONFIG.scale_factor == 0.001)
    SESSION_CONFIG.reset()
    MINI_CHECK(not SESSION_CONFIG.explode_mesh_faces)
    MINI_CHECK(SESSION_CONFIG.scale_factor == 1.0)


if __name__ == "__main__":
    run_all("python")
