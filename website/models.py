from django.db import models
from django.contrib.auth.hashers import make_password

class User(models.Model):
    user_id = models.AutoField(primary_key=True, db_column='user_ID')
    email = models.CharField(max_length=256, unique=True)
    username = models.CharField(max_length=150, unique=True)
    phone_number = models.CharField(max_length=256)
    password = models.CharField(max_length=256)  # Храним хеш
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)

    def set_password(self, raw_password):
        self.password = make_password(raw_password)

    def __str__(self):
        return self.username

    class Meta:
        db_table = 'users'
class Video(models.Model):
    status = models.BooleanField(default=False)
    author = models.CharField(max_length=255)
    source_name = models.CharField(max_length=255, blank=True)  # канал или имя источника
    title = models.CharField(max_length=255)
    url = models.URLField(blank=True)
    description = models.TextField(blank=True)
    uploaded_at = models.DateField(auto_now_add=True)
    video_path = models.CharField(max_length=500)  # путь к видеофайлу
    duration = models.PositiveIntegerField(help_text="Длительность в секундах", default=0)

    def __str__(self):
        return self.title

class VideoOwnership(models.Model):
    user = models.ForeignKey(User, on_delete=models.CASCADE)
    video = models.ForeignKey(Video, on_delete=models.CASCADE)

    class Meta:
        unique_together = ('user', 'video')

class Audio(models.Model):
    video = models.ForeignKey(Video, on_delete=models.CASCADE, related_name='audios')
    audio_path = models.CharField(max_length=500)  # путь к аудиофайлу
    transcription_path = models.CharField(max_length=500)  # путь к .txt с транскриптом

    def get_transcription_text(self):
        try:
            with open(self.transcription_path, 'r', encoding='utf-8') as f:
                return f.read()
        except Exception:
            return None

class Summary(models.Model):
    audio = models.ForeignKey(Audio, on_delete=models.CASCADE, related_name='summaries')
    file_path = models.CharField(max_length=500)  # путь к файлу с конспектом
    created_at = models.DateField(auto_now_add=True)
    total_rating = models.FloatField(default=0.0)
    format = models.CharField(max_length=50)

    def get_file_text(self):
        try:
            with open(self.file_path, 'r', encoding='utf-8') as f:
                return f.read()
        except Exception:
            return None

class SummaryReview(models.Model):
    summary = models.ForeignKey(Summary, on_delete=models.CASCADE, related_name='reviews')
    user = models.ForeignKey(User, on_delete=models.CASCADE, related_name='summary_reviews')
    created_at = models.DateField(auto_now_add=True)
    text = models.TextField()
    user_rating = models.PositiveSmallIntegerField()

    class Meta:
        unique_together = ('summary', 'user')

class Tag(models.Model):
    tag_name = models.CharField(max_length=100, unique=True)

    def __str__(self):
        return self.tag_name

class VideoTag(models.Model):
    video = models.ForeignKey(Video, on_delete=models.CASCADE)
    tag = models.ForeignKey(Tag, on_delete=models.CASCADE)

    class Meta:
        unique_together = ('video', 'tag')
