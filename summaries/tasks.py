from __future__ import annotations

from celery import shared_task


@shared_task(bind=True)
def process_lecture_note(self, note_id: int):
    from .services.pipeline import run_pipeline_for_note

    return run_pipeline_for_note(note_id=note_id, celery_task=self)
