from django.urls import path
from . import views

urlpatterns = [
    path("", views.research_list),
    path("upload_research", views.upload_research),
    path("view/<int:id>", views.view_research),
]
