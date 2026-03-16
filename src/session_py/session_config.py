class SessionConfig:
    _instance = None

    def __new__(cls):
        if not cls._instance:
            cls._instance = object.__new__(cls)
            cls._instance.explode_mesh_faces = False
        return cls._instance

    def reset(self):
        self.explode_mesh_faces = False

    def __repr__(self):
        return f"SessionConfig(explode_mesh_faces={self.explode_mesh_faces})"


SESSION_CONFIG = SessionConfig()
