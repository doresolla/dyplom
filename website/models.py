from django.db import models
from django.contrib.auth.hashers import make_password
import os

class User(models.Model):
    user_id = models.AutoField(primary_key=True, db_column='user_ID')
    email = models.EmailField(max_length=254, unique=True)
    username = models.CharField(max_length=150, unique=True)
    phone_number = models.CharField(max_length=256)
    password = models.CharField(max_length=256)  # Храним хеш
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)
    def __str__(self):
        return self.username
    class Meta:
        db_table = 'users'
    def edit_profile(self, username=None, email=None, phone_number=None):
        if username: self.username = username
        if email: self.email = email
        if phone_number: self.phone_number = phone_number
        self.save()
    def set_password(self, raw_password):
        self.password = make_password(raw_password)
    def delete_profile(self):
        self.delete()

class Video(models.Model):
    status = models.BooleanField(default=False)
    author = models.ForeignKey('User', on_delete=models.SET_NULL, null=True, blank=True, related_name='videos')
    source_name = models.CharField(max_length=255, blank=True)  # канал или имя источника
    title = models.CharField(max_length=255)
    url = models.URLField(blank=True)
    language = models.CharField(max_length=50, blank=True)
    description = models.TextField(blank=True)
    uploaded_at = models.DateField(auto_now_add=True)
    video_path = models.CharField(max_length=500)  # путь к видеофайлу
    duration = models.PositiveIntegerField(help_text="Длительность в секундах", default=0)

    def __str__(self):
        return self.title


    def get_status(self):
        return 'Обработан' if self.status else 'В обработке'


class VideoOwnership(models.Model):
    user = models.ForeignKey('User', on_delete=models.SET_NULL, null=True, blank=True)
    video = models.ForeignKey(Video, on_delete=models.CASCADE)

    class Meta:
        unique_together = ('user', 'video')


class Audio(models.Model):
    video = models.OneToOneField(Video, on_delete=models.CASCADE, related_name='audio')
    audio_path = models.CharField(max_length=500)  # путь к аудиофайлу
    transcription_path = models.CharField(max_length=500)  # путь к .txt с транскриптом
    error_message = models.TextField(null=True, blank=True)
    def get_transcription_text(self):
        try:
            print(f'self.transcription_path = {self.transcription_path}')
            with open(self.transcription_path, 'r', encoding='utf-8') as f:
                return f.read()
        except Exception:
            return None


class Format(models.Model):
    format_id = models.AutoField(primary_key=True)
    format = models.CharField(max_length=50, unique=True)

    def __str__(self):
        return self.format
    @classmethod
    def add_format(cls, format_str):
        return cls.objects.get_or_create(format_str=format_str)[0]

    def delete_format(self):
        self.delete()



class Algo(models.Model):
    algo_id = models.AutoField(primary_key=True)
    algo = models.CharField(max_length=100, unique=True)

    def __str__(self):
        return self.algo

    @classmethod
    def add_algo(cls, algo_str):
        return cls.objects.get_or_create(algo_str=algo_str)[0]

    def delete_algo(self):
        self.delete()

class Summary(models.Model):
    audio = models.ForeignKey(Audio, on_delete=models.CASCADE, related_name='summaries')
    user = models.ForeignKey('User', on_delete=models.CASCADE, related_name='summaries')
    file_path = models.CharField(max_length=500)  # путь к файлу с конспектом
    created_at = models.DateField(auto_now_add=True)
    total_rating = models.FloatField(default=0.0)
    format = models.ForeignKey(Format, on_delete=models.SET_NULL, null=True, related_name='summaries')
    algorithm = models.ForeignKey(Algo, on_delete=models.SET_NULL, null=True, related_name='summaries')

    def get_file_text(self):
        try:
            print(f'self.file_path={self.file_path}')
            txt_filename = os.path.splitext(self.file_path)[0] + ".txt"
            with open(txt_filename, 'r', encoding='utf-8') as f:
                return f.read()
        except Exception as e:
            print(f'Не удалось получить файл конспекта: {e}')
            return None

    def set_format(self, format_obj):
        self.format = format_obj
        self.save()

    def download(self):
        # Возвращаем путь к скачиванию, если существует
        import os
        return self.file_path if self.file_path and os.path.exists(self.file_path) else None

    def delete(self):
        import os
        if self.file_path and os.path.exists(self.file_path):
            os.remove(self.file_path)
        super().delete()

    def get_file_path(self):
        return self.file_path


class SummaryReview(models.Model):
    summary = models.ForeignKey(Summary, on_delete=models.CASCADE, related_name='reviews')
    user = models.ForeignKey(User, on_delete=models.CASCADE)
    created_at = models.DateTimeField(auto_now_add=True)
    text = models.TextField(blank=True)
    user_rating = models.IntegerField(default=0)

    def set_user_rating(self, rating):
        self.user_rating = rating
        self.save()

    def get_user_rating(self):
        return self.user_rating

    def set_text(self, text):
        self.text = text
        self.save()

    def get_text(self):
        return self.text

    @staticmethod
    def create_review(user, summary, text, user_rating):
        return SummaryReview.objects.create(user=user, summary=summary, text=text, user_rating=user_rating)

    def delete_review(self):
        self.delete()

    def edit_review(self, text=None, user_rating=None):
        if text is not None:
            self.text = text
        if user_rating is not None:
            self.user_rating = user_rating
        self.save()

    def __str__(self):
        return f"Отзыв {self.review_id} от {self.user.username} на Summary {self.summary_id}"
class Tag(models.Model):
    tag_name = models.CharField(max_length=100, unique=True)

    def __str__(self):
        return self.tag_name

    @classmethod
    def add_tag(cls, tag_name):
        return cls.objects.get_or_create(tag_name=tag_name)[0]

    def delete_tag(self):
        self.delete()

class VideoTag(models.Model):
    video = models.ForeignKey(Video, on_delete=models.CASCADE)
    tag = models.ForeignKey(Tag, on_delete=models.CASCADE)

    class Meta:
        unique_together = ('video', 'tag')

    def remove_tag(self):
        self.delete()

    def get_tag(self):
        return self.tag

    def set_tag(self, tag_obj):
        self.tag = tag_obj
        self.save()
