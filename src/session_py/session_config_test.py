from .mini_test import MINI_TEST
from .mini_test import MINI_CHECK
from .mini_test import run_all
from session_py.session_config import SESSION_CONFIG, SessionConfig


@MINI_TEST("SessionConfig", "Default Values")
def test_session_config_default_values():
    MINI_CHECK(SESSION_CONFIG.explode_mesh_faces == False)


@MINI_TEST("SessionConfig", "Runtime Modification")
def test_session_config_runtime_modification():
    MINI_CHECK(SESSION_CONFIG.explode_mesh_faces == False)
    SESSION_CONFIG.explode_mesh_faces = True
    MINI_CHECK(SESSION_CONFIG.explode_mesh_faces == True)
    SESSION_CONFIG.reset()
    MINI_CHECK(SESSION_CONFIG.explode_mesh_faces == False)


if __name__ == "__main__":
    run_all("python")
