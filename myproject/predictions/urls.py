from django.urls import path, re_path
from . import views

# Create your views here.
urlpatterns = [path("", views.index,name="index"),
               path("indian_stock", views.indian_stock,name="indian_stock"),
               path("us_stocks", views.us_stocks,name="us_stocks"),
               path("news", views.news,name="news"),
               path("predicted_stock", views.prediction_stock,name="predicted_stock"),
               path("predicted_stock_apple", views.prediction_stock_apple,name="predicted_stock_apple")

]