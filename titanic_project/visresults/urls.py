from django.urls import path
from . import views


app_name = 'visresults'

urlpatterns = [
    path('', views.home_page, name="home"),
    path('data/', views.data_explainer, name="dataset"),
    path('vis/', views.results_chart, name='results_chart'),

]
