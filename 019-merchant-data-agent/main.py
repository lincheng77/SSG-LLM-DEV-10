import uvicorn
from fastapi import FastAPI,Request
import uuid
from app.api.routers.query_router import query_router
from app.core.lifespan import lifespan
from app.core.context import set_req_id

# 创建应用对象
app = FastAPI(lifespan=lifespan)

# 注册路由
app.include_router(query_router)


@app.middleware("http")
async def add_process_time_header(request: Request, call_next):

    set_req_id(str(uuid.uuid4()))


    return await call_next(request)


if __name__ == '__main__':
    uvicorn.run(app, host="0.0.0.0", port=8000)

