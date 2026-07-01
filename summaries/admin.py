from django.contrib import admin

from .models import LectureNote, Notification, ProcessingLog, Review, Tag


class ProcessingLogInline(admin.TabularInline):
    model = ProcessingLog
    extra = 0
    readonly_fields = ('created_at', 'message')
    can_delete = False


@admin.register(Tag)
class TagAdmin(admin.ModelAdmin):
    list_display = ('name', 'slug', 'is_active', 'created_at')
    list_filter = ('is_active', 'created_at')
    search_fields = ('name', 'slug', 'description')
    prepopulated_fields = {'slug': ('name',)}


@admin.register(LectureNote)
class LectureNoteAdmin(admin.ModelAdmin):
    list_display = ('id', 'title', 'owner', 'status', 'progress', 'llm_model', 'is_favorite', 'created_at', 'completed_at')
    list_filter = ('status', 'llm_model', 'is_favorite', 'tags', 'created_at')
    search_fields = ('title', 'owner__username', 'summary_text', 'tags__name')
    readonly_fields = ('created_at', 'updated_at', 'completed_at')
    filter_horizontal = ('tags',)
    inlines = [ProcessingLogInline]


@admin.register(Notification)
class NotificationAdmin(admin.ModelAdmin):
    list_display = ('title', 'user', 'note', 'is_read', 'created_at')
    list_filter = ('is_read', 'created_at')
    search_fields = ('title', 'message', 'user__username')


@admin.register(Review)
class ReviewAdmin(admin.ModelAdmin):
    list_display = ('note', 'user', 'score', 'updated_at')
    list_filter = ('score', 'updated_at')
    search_fields = ('note__title', 'user__username', 'comment')


@admin.register(ProcessingLog)
class ProcessingLogAdmin(admin.ModelAdmin):
    list_display = ('note', 'created_at', 'short_message')
    search_fields = ('note__title', 'message')

    @admin.display(description='Сообщение')
    def short_message(self, obj):
        return obj.message[:120]
