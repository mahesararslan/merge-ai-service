"""Routes package initialization."""

from app.routes.health import router as health_router
from app.routes.ingest import router as ingest_router
from app.routes.query import router as query_router
from app.routes.study_plan import router as study_plan_router
from app.routes.utils import router as utils_router
from app.routes.vectors import router as vectors_router

__all__ = [
    "health_router",
    "ingest_router",
    "query_router",
    "study_plan_router",
    "utils_router",
    "vectors_router",
]
