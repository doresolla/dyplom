from __future__ import annotations

from django.conf import settings
from django.core.validators import MaxValueValidator, MinValueValidator
from django.db import models
from django.urls import reverse
from django.utils import timezone


class Tag(models.Model):
    """Справочник тегов. Наполняется администратором и используется в личных библиотеках."""

    name = models.CharField('Название тега', max_length=80, unique=True)
    slug = models.SlugField('Slug', max_length=100, unique=True)
    description = models.CharField('Описание', max_length=255, blank=True)
    is_active = models.BooleanField('Доступен пользователям', default=True)
    created_at = models.DateTimeField('Создан', default=timezone.now)

    class Meta:
        verbose_name = 'Тег'
        verbose_name_plural = 'Теги'
        ordering = ['name']

    def __str__(self) -> str:
        return self.name


class LectureNoteQuerySet(models.QuerySet):
    def owned_by(self, user):
        if not getattr(user, 'is_authenticated', False):
            return self.none()
        return self.filter(owner=user)


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
    tags = models.ManyToManyField(Tag, blank=True, related_name='lecture_notes', verbose_name='Теги')
    is_favorite = models.BooleanField('В избранном', default=False)
    video_file_original_name = models.CharField('Исходное имя видеофайла', max_length=255, blank=True)
    video_file = models.FileField('Видеофайл', upload_to='videos/%Y/%m/%d/', blank=True)
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
    created_at = models.DateTimeField('Создан', default=timezone.now)
    updated_at = models.DateTimeField('Обновлён', auto_now=True)
    completed_at = models.DateTimeField('Завершён', null=True, blank=True)

    objects = LectureNoteQuerySet.as_manager()

    class Meta:
        verbose_name = 'Конспект'
        verbose_name_plural = 'Конспекты'
        ordering = ['-created_at']
        indexes = [
            models.Index(fields=['owner', '-created_at'], name='note_owner_created_idx'),
            models.Index(fields=['owner', 'is_favorite'], name='note_owner_fav_idx'),
        ]

    def __str__(self) -> str:
        return self.title

    def get_absolute_url(self):
        return reverse('note_detail', kwargs={'pk': self.pk})


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


class Review(models.Model):
    """Личный отзыв владельца о качестве сформированного для него конспекта."""

    note = models.OneToOneField(LectureNote, on_delete=models.CASCADE, related_name='review', verbose_name='Конспект')
    user = models.ForeignKey(settings.AUTH_USER_MODEL, on_delete=models.CASCADE, related_name='note_reviews', verbose_name='Пользователь')
    score = models.PositiveSmallIntegerField('Оценка', validators=[MinValueValidator(1), MaxValueValidator(5)])
    comment = models.TextField('Комментарий', blank=True)
    created_at = models.DateTimeField('Создан', default=timezone.now)
    updated_at = models.DateTimeField('Обновлён', auto_now=True)

    class Meta:
        verbose_name = 'Отзыв'
        verbose_name_plural = 'Отзывы'
        ordering = ['-updated_at']

    def __str__(self) -> str:
        return f'{self.note.title}: {self.score}/5'
