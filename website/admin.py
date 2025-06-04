from django.contrib import admin
from .models import User, Video, VideoOwnership, Audio, Summary, SummaryReview, Tag, VideoTag
from django.contrib.auth.admin import UserAdmin as BaseUserAdmin


@admin.register(User)
class UserAdmin(admin.ModelAdmin):
    list_display = ('user_id', 'email', 'username', 'phone_number', 'created_at')
    search_fields = ('email', 'username')
    readonly_fields = ('created_at', 'updated_at')
@admin.register(Video)
class VideoAdmin(admin.ModelAdmin):
    list_display = ('id', 'title', 'author', 'uploaded_at', 'status')
    list_filter = ('status', 'uploaded_at')
    search_fields = ('title', 'description')


@admin.register(VideoOwnership)
class VideoOwnershipAdmin(admin.ModelAdmin):
    list_display = ('user', 'video')


@admin.register(Audio)
class AudioAdmin(admin.ModelAdmin):
    list_display = ('id', 'video', 'audio_path')


@admin.register(Summary)
class SummaryAdmin(admin.ModelAdmin):
    list_display = ('id', 'audio', 'format', 'total_rating', 'created_at')


@admin.register(SummaryReview)
class SummaryReviewAdmin(admin.ModelAdmin):
    list_display = ('id', 'summary', 'user', 'user_rating', 'created_at')


@admin.register(Tag)
class TagAdmin(admin.ModelAdmin):
    list_display = ('id', 'tag_name')
    search_fields = ('tag_name',)


@admin.register(VideoTag)
class VideoTagAdmin(admin.ModelAdmin):
    list_display = ('video', 'tag')