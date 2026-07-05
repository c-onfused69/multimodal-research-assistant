"""FastAPI middleware to inject trace IDs."""
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from observability.tracer import Tracer


class TraceMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):
        tracer = Tracer.get()
        # Create a trace for the request
        trace = tracer.trace(
            name=f"{request.method} {request.url.path}",
            tags=["api"]
        )
        # Store in request state
        request.state.trace = trace
        
        response = await call_next(request)
        
        # Flush traces (async non-blocking in langfuse-python)
        tracer.flush()
        return response
