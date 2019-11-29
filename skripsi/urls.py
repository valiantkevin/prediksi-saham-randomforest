from django.contrib import admin
from django.urls import include, path

urlpatterns = [
    path('prediksi/', include('prediksi.urls')),
    path('admin/', admin.site.urls),
]
