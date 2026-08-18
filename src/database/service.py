"""Application-facing facade; Hybrid remains unaware of SQL."""
from src.database import repository as repo

def persist_student_context(*args, **kwargs): return repo.upsert_student(*args, **kwargs)
def persist_prediction(*args, **kwargs): return repo.upsert_prediction(*args, **kwargs)
def load_latest_prediction(enrollment_id): return repo.get_latest_prediction(enrollment_id)
def load_student_timeline(enrollment_id):
    from sqlalchemy import text
    from src.database.connection import engine
    with engine.connect() as c: return c.execute(text("SELECT * FROM data.temporal_observation WHERE enrollment_id=:e ORDER BY time_index"), {"e":enrollment_id}).mappings().all()
def save_hybrid_prediction(**kwargs): return repo.upsert_prediction(**kwargs)
