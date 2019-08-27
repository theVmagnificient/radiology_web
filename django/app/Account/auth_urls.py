from django.urls import path
from . import auth as views

urlpatterns = [
    path('', views.auth, name='auth'),
    path('login/', views.login, name='login'),
    path('logout/', views.deAuth, name='logout'),
]