from django.contrib import admin
from django.urls import path
from django.conf.urls import include
from . import views
from django.conf.urls.static import static
from Frontend import settings
from Account import auth as Account_views
from Account import profile_views

urlpatterns = [
	path('', views.redirect),
    path('landpage/', Account_views.landpage),
    path('feedback/', Account_views.feedback),
    path('admin/', admin.site.urls),
    path('auth/', include('Account.auth_urls')),
    path('home/', include('Account.home_urls')),
    path('series/', include('Slicer.urls')),
    path('profile/<str:login>', profile_views.viewProfile),
    # path('check_error_page/', Account_views.check_error_page),
]

urlpatterns += static(settings.STATIC_URL, document_root=settings.STATIC_ROOT)
urlpatterns += static(settings.MEDIA_URL, document_root=settings.MEDIA_ROOT)

handler404 = Account_views.error_404
handler500 = Account_views.error_500