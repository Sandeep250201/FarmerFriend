from django.urls import path
from .views import crop_prediction_page, predict_crop, predict_crop1, predict_yield, predict_crops, get_states_and_districts, get_states, get_districts, recommend_profitable_crops

urlpatterns = [
    path("", crop_prediction_page, name="crop_prediction_page"),
    path("predict/", predict_crop, name="predict_crop"),
    path("get-states-districts/", get_states_and_districts, name= "get_states_and_districts"),
    path("predict-location/", predict_crop1, name="predict_crop"),
    path('predict-crop/', predict_crops, name='predict_crops'),
    path('predict-yield/', predict_yield, name='predict_yield'),
    path("get-states/", get_states, name="get_states"),
    path("get-districts/", get_districts, name="get_districts"),
    path("recommend_profitable_crop/", recommend_profitable_crops, name="recommend_profitable_crop"),
]
