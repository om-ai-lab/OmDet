import time
import uvicorn
from fastapi import FastAPI
from omdet.inference.det_engine import DetEngine
from pydantic import BaseModel
from typing import List, Dict, Union


class InfDetectBody(BaseModel):
    model_id: str
    data: List[str]
    src_type: str = "url"
    task: str
    labels: List[str]
    threshold: float = 0.1
    nms_threshold: float = 0.5


class Object(BaseModel):
    xmin: float
    ymin: float
    xmax: float
    ymax: float
    conf: float
    label: str


class DetectionRes(BaseModel):
    took: int
    objects: List[List[Object]] = []


app = FastAPI()


@app.on_event("startup")
async def startup_event():
    app.state.detector = DetEngine(model_dir="resources/", device="cuda", batch_size=10)


@app.post(
    "/inf_predict",
    response_model=DetectionRes,
    name="Detect objects with Inf Possibilities",
)
async def detect_urls(
        body: InfDetectBody = None,
) -> DetectionRes:
    s_time = time.time()
    out = app.state.detector.inf_predict(
        body.model_id,
        task=body.task,
        labels=body.labels,
        data=body.data,
        src_type=body.src_type,
        conf_threshold=body.threshold,
        nms_threshold=body.nms_threshold,
    )

    resp = DetectionRes(took=int((time.time() - s_time) * 1000), objects=out)
    return resp


if __name__ == "__main__":
    uvicorn.run("run_wsgi:app", host="0.0.0.0", port=8000)
