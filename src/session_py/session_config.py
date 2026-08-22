from __future__ import annotations
class SessionConfig:
    _instance = None

    def __new__(cls):
        if not cls._instance:
            cls._instance = object.__new__(cls)
            cls._instance.explode_mesh_faces = False
            cls._instance.scale_factor = 1.0
        return cls._instance

    def reset(self) -> None:
        self.explode_mesh_faces = False
        self.scale_factor = 1.0

    def __repr__(self):
        return f"SessionConfig(explode_mesh_faces={self.explode_mesh_faces}, scale_factor={self.scale_factor})"


SESSION_CONFIG = SessionConfig()
