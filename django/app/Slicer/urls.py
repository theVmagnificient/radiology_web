from django.urls import path
from . import views

urlpatterns = [
    path("", views.research_list),
    path("upload_research", views.upload_research),
    path("view/<int:id>", views.view_research),
    path("mark_up/<int:id>", views.mark_up_research),
    path("kafka_processed", views.kafka_processed),
]
