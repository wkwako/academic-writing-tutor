from django.urls import path
from . import views

urlpatterns = [
    path("", views.tutor_page, name="tutor_page"),
]