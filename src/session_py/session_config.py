from __future__ import annotations


class SessionConfig:
    """Runtime settings used by session operations."""

    explode_mesh_faces: bool  # Whether meshing emits one face per triangle.
    scale_factor: float  # Scale applied by external session adapters.

    def __init__(self) -> None:
        """Creates settings with their default values."""

        self.explode_mesh_faces = False
        self.scale_factor = 1.0

    def reset(self) -> None:
        """Restores every setting to its default value."""

        self.explode_mesh_faces = False
        self.scale_factor = 1.0

    def __repr__(self) -> str:
        """Returns a constructor-style representation."""
        return f"SessionConfig(explode_mesh_faces={self.explode_mesh_faces}, scale_factor={self.scale_factor})"


SESSION_CONFIG: SessionConfig = SessionConfig()  # Process-wide settings used by session operations.
