from django.contrib import admin
from .models import User, Feedback, ExtendedUser, Chat, News

admin.site.register(User)
admin.site.register(Feedback)
admin.site.register(Chat)
admin.site.register(News)

@admin.register(ExtendedUser)
class ExtendedUserAdmin(admin.ModelAdmin):
    readonly_fields = ('ava',)
