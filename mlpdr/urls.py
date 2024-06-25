
from django.contrib import admin
from django.urls import path
from .views import process_license_plate

urlpatterns = [
    #path('', process_license_plate, name='home'),  # URL racine
    path('process_license_plate/', process_license_plate, name='process_license_plate'),
    path('admin/', admin.site.urls),
]


