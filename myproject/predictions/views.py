# views.py

from django.shortcuts import render
from .stock import predict_stock_price
from .apple import predict_stock_price_apple

def index(request):
    return render(request, 'index.html')

def indian_stock(request):
    return render(request, 'indian_stock.html')

def us_stocks(request):
    return render(request, 'us_stocks.html')

def news(request):
    return render(request, 'news.html')

def prediction_stock(request):
    try:
        prediction_result = predict_stock_price()
        return render(request, 'predicted_stock.html',{'prediction_result': prediction_result})
    except Exception as e:
        error_message = str(e)
        return render(request, 'prediction_stock.html', {'error_message': error_message})

def prediction_stock_apple(request):
    try:
        # prediction_result = predict_stock_price_apple()
        return render(request, 'apple.html', )
        # {'prediction_result_apple': prediction_result}
    except Exception as e:
        error_message = str(e)
        return render(request, 'apple.html', {'error_message': error_message})
