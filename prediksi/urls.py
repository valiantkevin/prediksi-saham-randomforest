from django.urls import path

from . import views

urlpatterns = [
    path('', views.index, name='home'),
    path('update/', views.update, name='update'),
    path('do_update/', views.do_update, name='do_update'),
    path('data/', views.data_main, name='data_main'),
    path('<str:ticker>/data/', views.data, name='data'),
    path('chart/', views.chart_main, name='chart_main'),
    path('<str:ticker>/chart/', views.chart, name='chart'),
    path('result/', views.result_main, name='result_main'),
    path('<str:ticker>/result/<str:d>/', views.result, name='result'),
    path('summary/', views.summary, name='summary'),
    path('train/', views.train, name='train'),
    path('tree/', views.tree_main, name='tree_main'),
    path('tree/<str:ticker>/<str:tipe>/<str:jarak>/<str:number>', views.tree, name='tree'),
    path('<str:ticker>/coba/', views.coba, name='coba'),
]