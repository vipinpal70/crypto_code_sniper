from fastapi import FastAPI, HTTPException, Depends

from celery import Celery

app = FastAPI()

# celery Configuration

celery = Celery(
    "tasks",
    broker="redis://localhost:6379/0",
    backend='backend://redis://localhost:6379/0'
    
    )


@celery.task
def schedule_task(task_id=None):
    if task_id is None:
        raise ValueError("Task ID must be provided")
    # Simulate a long-running task
    import time
    time.sleep(10)
    return f"Task {task_id} completed"


@app.post("/schedule_task/{task_id}")
async def schedule_task_endpoint(task_id: str):
    try:
        task = schedule_task.apply_async(args=[task_id])
        return {"task_id": task.id, "status": "Task scheduled"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

