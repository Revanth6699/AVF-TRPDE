from __future__ import annotations

from fastapi import FastAPI

from backend.app.api.v1.datasets import router as datasets_router
from backend.app.api.v1.experiments import router as experiments_router
from backend.app.api.v1.forecasts import router as forecasts_router
from backend.app.api.v1.risk import router as risk_router
from backend.app.api.v1.portfolios import router as portfolios_router
from backend.app.api.v1.reports import router as reports_router


app = FastAPI(
    title="AVF-TRPDE",
    description=(
        "Adaptive Volatility Forecasting, Tail-Risk & "
        "Portfolio Decision Engine"
    ),
    version="1.0.0",
)


app.include_router(datasets_router)
app.include_router(experiments_router)
app.include_router(forecasts_router)
app.include_router(risk_router)
app.include_router(portfolios_router)
app.include_router(reports_router)


@app.get("/health", tags=["System"])
def health_check() -> dict[str, str]:
    """Return application health status."""

    return {
        "status": "ok",
        "service": "AVF-TRPDE",
    }