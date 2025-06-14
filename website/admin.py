from django.contrib import admin
from .models import (
    User,
    Video,
    VideoOwnership,
    Audio,
    Summary,
    SummaryReview,
    Tag,
    VideoTag,
    Format,
    Algo
)


@admin.register(User)
class UserAdmin(admin.ModelAdmin):
    list_display = ('user_id', 'email', 'username', 'phone_number', 'created_at')
    search_fields = ('email', 'username')
    readonly_fields = ('created_at', 'updated_at')


@admin.register(Video)
class VideoAdmin(admin.ModelAdmin):
    list_display = ('id', 'title', 'author', 'uploaded_at', 'status', 'duration')
    list_filter = ('status', 'uploaded_at', 'language')
    search_fields = ('title', 'description', 'source_name')


@admin.register(VideoOwnership)
class VideoOwnershipAdmin(admin.ModelAdmin):
    list_display = ('user', 'video')


@admin.register(Audio)
class AudioAdmin(admin.ModelAdmin):
    list_display = ('id', 'video', 'audio_path', 'transcription_path')


@admin.register(Summary)
class SummaryAdmin(admin.ModelAdmin):
    list_display = ('id', 'audio', 'user', 'format', 'algorithm', 'total_rating', 'created_at')
    list_filter = ('format', 'algorithm', 'created_at')
    search_fields = ('file_path',)


@admin.register(SummaryReview)
class SummaryReviewAdmin(admin.ModelAdmin):
    list_display = ['id', 'user', 'summary', 'user_rating', 'created_at']
    search_fields = ['user__username', 'summary__audio__video__title']
    list_filter = ['user_rating', 'created_at']


@admin.register(Tag)
class TagAdmin(admin.ModelAdmin):
    list_display = ('id', 'tag_name')
    search_fields = ('tag_name',)


@admin.register(VideoTag)
class VideoTagAdmin(admin.ModelAdmin):
    list_display = ('video', 'tag')


@admin.register(Format)
class FormatAdmin(admin.ModelAdmin):
    list_display = ('format_id', 'format')
    search_fields = ('format',)


@admin.register(Algo)
class AlgoAdmin(admin.ModelAdmin):
    list_display = ('algo_id', 'algo')
    search_fields = ('algo',)
