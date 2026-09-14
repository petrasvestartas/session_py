from .mini_test import MINI_TEST
from .mini_test import MINI_CHECK
from .mini_test import run_all


@MINI_TEST("SessionConfig", "Runtime Modification")
def test_session_config_runtime_modification():
    from session_py import SESSION_CONFIG
    from session_py import SessionConfig

    SESSION_CONFIG.reset()
    config = SessionConfig()
    other = SessionConfig()

    MINI_CHECK(not config.explode_mesh_faces)
    MINI_CHECK(config.scale_factor == 1.0)
    config.explode_mesh_faces = True
    config.scale_factor = 0.001
    MINI_CHECK(config.explode_mesh_faces)
    MINI_CHECK(config.scale_factor == 0.001)
    MINI_CHECK(not other.explode_mesh_faces)
    MINI_CHECK(other.scale_factor == 1.0)
    MINI_CHECK(not SESSION_CONFIG.explode_mesh_faces)
    MINI_CHECK(SESSION_CONFIG.scale_factor == 1.0)
    SESSION_CONFIG.explode_mesh_faces = True
    SESSION_CONFIG.scale_factor = 0.001
    MINI_CHECK(SESSION_CONFIG.explode_mesh_faces)
    MINI_CHECK(SESSION_CONFIG.scale_factor == 0.001)
    SESSION_CONFIG.reset()
    MINI_CHECK(not SESSION_CONFIG.explode_mesh_faces)
    MINI_CHECK(SESSION_CONFIG.scale_factor == 1.0)
    config.reset()
    MINI_CHECK(not config.explode_mesh_faces)
    MINI_CHECK(config.scale_factor == 1.0)


if __name__ == "__main__":
    run_all("python")
