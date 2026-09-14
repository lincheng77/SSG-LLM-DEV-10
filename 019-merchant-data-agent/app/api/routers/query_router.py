from fastapi import APIRouter, Depends
from starlette.responses import StreamingResponse

from app.api.dependencies import get_query_service
from app.api.schemas.query_schema import QuerySchema
from app.service.query_service import QueryService

query_router = APIRouter()


@query_router.post("/api/query")
async def search(query_schema: QuerySchema, service: QueryService=Depends(get_query_service)):

    result = service.search(query_schema.query)

    return StreamingResponse(result,media_type="text/event-stream")
