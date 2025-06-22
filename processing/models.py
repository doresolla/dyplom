# -*- coding: utf-8 -*-
from django.db import models
from website.models import User, Video, Audio, Summary

class ProcessingRequest(models.Model):
    user = models.ForeignKey(User, on_delete=models.CASCADE)
    video = models.ForeignKey(Video, on_delete=models.CASCADE)
    audio = models.ForeignKey(Audio, on_delete=models.CASCADE)
    summary = models.OneToOneField(Summary, on_delete=models.CASCADE, null=True, blank=True)
    algorithm = models.CharField(max_length=50)
    format = models.CharField(max_length=10)
    ratio = models.FloatField(default=0.5)
    requested_at = models.DateTimeField(auto_now_add=True)
    completed_at = models.DateTimeField(null=True, blank=True)

    def __str__(self):
        return f"Запрос №{self.id} от {self.user.username}"
