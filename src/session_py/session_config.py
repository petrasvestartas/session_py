from __future__ import annotations


class SessionConfig:
    """Runtime settings used by session operations.

    Attributes
    ----------
    explode_mesh_faces : bool
        Whether meshing emits one face per triangle.
    scale_factor : float
        Scale applied by external session adapters.

    Examples
    --------
    >>> config = SessionConfig()
    >>> config.scale_factor
    1.0
    >>> repr(config)
    'SessionConfig(explode_mesh_faces=False, scale_factor=1.0)'
    """

    explode_mesh_faces: bool
    scale_factor: float

    def __init__(self) -> None:
        self.explode_mesh_faces = False
        self.scale_factor = 1.0

    def reset(self) -> None:
        """Restore every setting to its default value."""
        self.explode_mesh_faces = False
        self.scale_factor = 1.0

    def __repr__(self) -> str:
        """Return a constructor-style representation."""
        return f"SessionConfig(explode_mesh_faces={self.explode_mesh_faces}, scale_factor={self.scale_factor})"


SESSION_CONFIG: SessionConfig = SessionConfig()
