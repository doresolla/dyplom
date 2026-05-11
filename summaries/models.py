from __future__ import annotations

from django.conf import settings
from django.core.validators import MaxValueValidator, MinValueValidator
from django.db import models
from django.db.models import Avg
from django.urls import reverse
from django.utils import timezone


class LectureNote(models.Model):
    class Status(models.TextChoices):
        QUEUED = 'queued', 'В очереди'
        PROCESSING = 'processing', 'Обрабатывается'
        DONE = 'done', 'Готово'
        ERROR = 'error', 'Ошибка'

    class LlmModel(models.TextChoices):
        QWEN = 'qwen', 'Qwen'
        MISTRAL = 'mistral', 'Mistral'

    owner = models.ForeignKey(
        settings.AUTH_USER_MODEL,
        on_delete=models.CASCADE,
        related_name='lecture_notes',
        verbose_name='Владелец',
    )
    title = models.CharField('Название', max_length=255)
    video_file = models.FileField('Видеофайл', upload_to='videos/%Y/%m/%d/')
    summary_file = models.FileField('Markdown-файл конспекта', upload_to='summaries/%Y/%m/%d/', blank=True)
    summary_pdf_file = models.FileField('PDF-файл конспекта', upload_to='summaries/%Y/%m/%d/', blank=True)
    summary_text = models.TextField('Текст конспекта', blank=True)
    transcript_file = models.FileField('Транскрипция', upload_to='transcripts/%Y/%m/%d/', blank=True)
    status = models.CharField('Статус', max_length=20, choices=Status.choices, default=Status.QUEUED)
    progress = models.PositiveSmallIntegerField('Прогресс, %', default=0)
    status_message = models.CharField('Сообщение статуса', max_length=500, blank=True)
    llm_model = models.CharField('LLM-модель', max_length=20, choices=LlmModel.choices, default=LlmModel.QWEN)
    error_message = models.TextField('Текст ошибки', blank=True)
    result_dir = models.CharField('Папка результата', max_length=1000, blank=True)
    pipeline_meta = models.JSONField('Метаданные пайплайна', default=dict, blank=True)
    is_public = models.BooleanField('Доступен всем', default=True)
    created_at = models.DateTimeField('Создан', default=timezone.now)
    updated_at = models.DateTimeField('Обновлен', auto_now=True)
    completed_at = models.DateTimeField('Завершен', null=True, blank=True)

    class Meta:
        verbose_name = 'Конспект'
        verbose_name_plural = 'Конспекты'
        ordering = ['-created_at']

    def __str__(self) -> str:
        return self.title

    def get_absolute_url(self):
        return reverse('note_detail', kwargs={'pk': self.pk})

    @property
    def average_rating(self) -> float | None:
        value = self.ratings.aggregate(avg=Avg('score'))['avg']
        return round(value, 2) if value is not None else None

    @property
    def ratings_count(self) -> int:
        return self.ratings.count()


class ProcessingLog(models.Model):
    note = models.ForeignKey(LectureNote, on_delete=models.CASCADE, related_name='logs')
    created_at = models.DateTimeField(default=timezone.now)
    message = models.TextField()

    class Meta:
        ordering = ['created_at']

    def __str__(self) -> str:
        return f'{self.note_id}: {self.message[:80]}'


class Notification(models.Model):
    user = models.ForeignKey(settings.AUTH_USER_MODEL, on_delete=models.CASCADE, related_name='notifications')
    note = models.ForeignKey(LectureNote, on_delete=models.CASCADE, related_name='notifications', null=True, blank=True)
    title = models.CharField(max_length=255)
    message = models.TextField()
    is_read = models.BooleanField(default=False)
    created_at = models.DateTimeField(default=timezone.now)

    class Meta:
        ordering = ['-created_at']

    def __str__(self) -> str:
        return self.title


class Rating(models.Model):
    note = models.ForeignKey(LectureNote, on_delete=models.CASCADE, related_name='ratings')
    user = models.ForeignKey(settings.AUTH_USER_MODEL, on_delete=models.CASCADE, related_name='summary_ratings')
    score = models.PositiveSmallIntegerField(validators=[MinValueValidator(1), MaxValueValidator(5)])
    comment = models.TextField(blank=True)
    created_at = models.DateTimeField(default=timezone.now)
    updated_at = models.DateTimeField(auto_now=True)

    class Meta:
        unique_together = ('note', 'user')
        ordering = ['-updated_at']

    def __str__(self) -> str:
        return f'{self.note_id}: {self.score}'
