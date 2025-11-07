"""
Vertex AI Veo video generation module.

This module provides support for Vertex AI Veo video generation models:
- veo-3.0-generate-001
- veo-3.0-fast-generate-001
- veo-3.0-generate-preview
- veo-3.0-fast-generate-preview
- veo-3.1-generate-preview
- veo-3.1-fast-generate-preview
"""

from litellm.llms.vertex_ai.video_generation.video_generation_handler import (
    VertexVideoGeneration,
)

__all__ = ["VertexVideoGeneration"]

