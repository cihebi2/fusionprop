from django.urls import path
from . import views

urlpatterns = [
    path('', views.submit_prediction, name='submit_prediction'),
    # Add the new URL pattern for the job list page
    path('jobs/', views.job_list, name='job_list'),
    path('result/<uuid:job_id>/', views.prediction_result, name='prediction_result'),
    path('status/<uuid:job_id>/', views.check_job_status, name='check_job_status'),
    path('result/<uuid:job_id>/download/', views.download_results, name='download_results'),
]