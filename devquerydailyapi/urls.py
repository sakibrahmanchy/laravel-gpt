"""devquerydailyapi URL Configuration

The `urlpatterns` list routes URLs to views. For more information please see:
    https://docs.djangoproject.com/en/3.2/topics/http/urls/
Examples:
Function views
    1. Add an import:  from my_app import views
    2. Add a URL to urlpatterns:  path('', views.home, name='home')
Class-based views
    1. Add an import:  from other_app.views import Home
    2. Add a URL to urlpatterns:  path('', Home.as_view(), name='home')
Including another URLconf
    1. Import the include() function: from django.urls import include, path
    2. Add a URL to urlpatterns:  path('blog/', include('blog.urls'))
"""
from django.contrib import admin
from django.urls import path,include
from rest_framework.routers import DefaultRouter
from .viewsets.task import TaskViewSet
from .viewsets.train import LoadUrlsAPI, AskAI, LoadTrainData, SuggestCodingProblem, LoadTrainDataWithAstraDB

router = DefaultRouter()
router.register(r'tasks', TaskViewSet)

urlpatterns = [
    path('admin/', admin.site.urls),
    path('load-url/', LoadUrlsAPI.as_view(), name='url-loader'),
    path('ask-ai/', AskAI.as_view(), name='ask-ai'),
    path('load-train-data/', LoadTrainData.as_view(), name='load-train-data'),
    path('load-train-data-astra/',LoadTrainDataWithAstraDB.as_view(), name='load-train-data'),
    path('coding-problem/', SuggestCodingProblem.as_view(), name='suggest-coding-problem'),
    path('', include(router.urls))
]
