from django.contrib.auth.views import LoginView, LogoutView
from django.urls import path
from . import views

urlpatterns = [
    # Home and main pages
    path('', views.home, name='home'),
    path('main/', views.main, name='main'),

    # User authentication
    path('register/', views.register_view, name='register'),
    path('login/', LoginView.as_view(template_name='login.html'), name='login'),
    path('logout/', LogoutView.as_view(next_page='/'), name='logout'),

    # Additional pages
    path('search/', views.search, name='search'),
    path('terms_service', views.terms_service, name='terms_service'),
    path('jobs/', views.job_list, name='job_list'),
    path('profiles/', views.profiles, name='profiles'),
]

