from sklearn.ensemble import RandomForestClassifier
from django.http import HttpResponse
import pandas as pd
import numpy as np
from ta import *
import os
from django.shortcuts import get_object_or_404, render, redirect
from .models import Stocks,PlainStocks,GroupedStocks,OneHotStocks
import datetime
from joblib import dump, load
from sklearn.tree import export_graphviz
import graphviz
import time

idx = ['ADRO', 'ADHI', 'AKRA', 'ANTM', 'ASII', 'BBCA', 'BBNI', 'BBRI', 'BBTN', 'BJBR', 'BKSL', 'BMRI', 'BDSE', 'ELSA', 'EXCL', 'GGRM', 'HMSP', 'ICBP', 'INCO', 'INDF', 'INKP', 'INTP', 'ITMG', 'JSMR', 'KLBF', 'LPKR', 'LPPF', 'MEDC', 'PGAS', 'PTBA', 'PTPP', 'SCMA', 'SMGR', 'SRIL', 'SSMS', 'TLKM', 'UNTR', 'UNVR', 'WIKA', 'WSKT']
parent = os.getcwd()
stat = parent +"\\prediksi\\static\\prediksi"


def index(request):
    if request.method == 'POST' and request.FILES['myfile']:
        myfile = request.FILES['myfile']
        data = pd.read_csv(myfile)
        ADRO = data[data['Ticker']=='ADRO']
        ADRO = ADRO.reset_index(drop=True)
        ADHI = data[data['Ticker']=='ADHI']
        ADHI = ADHI.reset_index(drop=True)
        AKRA = data[data['Ticker']=='AKRA']
        AKRA = AKRA.reset_index(drop=True)
        ANTM = data[data['Ticker']=='ANTM']
        ANTM = ANTM.reset_index(drop=True)
        ASII = data[data['Ticker']=='ASII']
        ASII = ASII.reset_index(drop=True)
        BBCA = data[data['Ticker']=='BBCA']
        BBCA = BBCA.reset_index(drop=True)
        BBNI = data[data['Ticker']=='BBNI']
        BBNI = BBNI.reset_index(drop=True)
        BBRI = data[data['Ticker']=='BBRI']
        BBRI = BBRI.reset_index(drop=True)
        BBTN = data[data['Ticker']=='BBTN']
        BBTN = BBTN.reset_index(drop=True)
        BJBR = data[data['Ticker']=='BJBR']
        BJBR = BJBR.reset_index(drop=True)
        BKSL = data[data['Ticker']=='BKSL']
        BKSL = BKSL.reset_index(drop=True)
        BMRI = data[data['Ticker']=='BMRI']
        BMRI = BMRI.reset_index(drop=True)
        #BRPT = data[data['Ticker']=='BRPT']
        #BRPT = BRPT.reset_index(drop=True)
        BDSE = data[data['Ticker']=='BDSE']
        BDSE = BDSE.reset_index(drop=True)
        ELSA = data[data['Ticker']=='ELSA']
        ELSA = ELSA.reset_index(drop=True)
        EXCL = data[data['Ticker']=='EXCL']
        EXCL = EXCL.reset_index(drop=True)
        GGRM = data[data['Ticker']=='GGRM']
        GGRM = GGRM.reset_index(drop=True)
        HMSP = data[data['Ticker']=='HMSP']
        HMSP = HMSP.reset_index(drop=True)
        ICBP = data[data['Ticker']=='ICBP']
        ICBP = ICBP.reset_index(drop=True)
        INCO = data[data['Ticker']=='INCO']
        INCO = INCO.reset_index(drop=True)
        INDF = data[data['Ticker']=='INDF']
        INDF = INDF.reset_index(drop=True)
        #INDY = data[data['Ticker']=='INDY']
        #INDY = INDY.reset_index(drop=True)
        INKP = data[data['Ticker']=='INKP']
        INKP = INKP.reset_index(drop=True)
        INTP = data[data['Ticker']=='INTP']
        INTP = INTP.reset_index(drop=True)
        ITMG = data[data['Ticker']=='ITMG']
        ITMG = ITMG.reset_index(drop=True)
        JSMR = data[data['Ticker']=='JSMR']
        JSMR = JSMR.reset_index(drop=True)
        KLBF = data[data['Ticker']=='KLBF']
        KLBF = KLBF.reset_index(drop=True)
        LPKR = data[data['Ticker']=='LPKR']
        LPKR = LPKR.reset_index(drop=True)
        LPPF = data[data['Ticker']=='LPPF']
        LPPF = LPPF.reset_index(drop=True)
        MEDC = data[data['Ticker']=='MEDC']
        MEDC = MEDC.reset_index(drop=True)
        #MNCN = data[data['Ticker']=='MNCN']
        #MNCN = MNCN.reset_index(drop=True)
        PGAS = data[data['Ticker']=='PGAS']
        PGAS = PGAS.reset_index(drop=True)
        PTBA = data[data['Ticker']=='PTBA']
        PTBA = PTBA.reset_index(drop=True)
        PTPP = data[data['Ticker']=='PTPP']
        PTPP = PTPP.reset_index(drop=True)
        SCMA = data[data['Ticker']=='SCMA']
        SCMA = SCMA.reset_index(drop=True)
        SMGR = data[data['Ticker']=='SMGR']
        SMGR = SMGR.reset_index(drop=True)
        SRIL = data[data['Ticker']=='SRIL']
        SRIL = SRIL.reset_index(drop=True)
        SSMS = data[data['Ticker']=='SSMS']
        SSMS = SSMS.reset_index(drop=True)
        TLKM = data[data['Ticker']=='TLKM']
        TLKM = TLKM.reset_index(drop=True)
        UNTR = data[data['Ticker']=='UNTR']
        UNTR = UNTR.reset_index(drop=True)
        UNVR = data[data['Ticker']=='UNVR']
        UNVR = UNVR.reset_index(drop=True)
        WIKA = data[data['Ticker']=='WIKA']
        WIKA = WIKA.reset_index(drop=True)
        #WSBP = data[data['Ticker']=='WSBP']
        #WSBP = WSBP.reset_index(drop=True)
        WSKT = data[data['Ticker']=='WSKT']
        WSKT = WSKT.reset_index(drop=True)
        data = {'ADRO' : ADRO, 'ADHI' : ADHI, 'AKRA' : AKRA, 'ANTM' : ANTM, 'ASII' : ASII, 'BBCA' : BBCA, 'BBNI' : BBNI, 'BBRI' : BBRI, 'BBTN' : BBTN, 'BJBR' : BJBR, 'BKSL' : BKSL, 'BMRI' : BMRI, 'BDSE' : BDSE, 'ELSA' : ELSA, 'EXCL' : EXCL, 'GGRM' : GGRM, 'HMSP' : HMSP, 'ICBP' : ICBP, 'INCO' : INCO, 'INDF' : INDF, 'INKP' : INKP, 'INTP' : INTP, 'ITMG' : ITMG, 'JSMR' : JSMR, 'KLBF' : KLBF, 'LPKR' : LPKR, 'LPPF' : LPPF, 'MEDC' : MEDC, 'PGAS' : PGAS, 'PTBA' : PTBA, 'PTPP' : PTPP, 'SCMA' : SCMA, 'SMGR' : SMGR, 'SRIL' : SRIL, 'SSMS' : SSMS, 'TLKM' : TLKM, 'UNTR' : UNTR, 'UNVR' : UNVR, 'WIKA' : WIKA, 'WSKT' : WSKT}
        #Stocks
        Stocks.objects.all().delete()
        PlainStocks.objects.all().delete()
        GroupedStocks.objects.all().delete()
        OneHotStocks.objects.all().delete()
        for key, value in data.items():
            for i in range(0, value.shape[0]-1):
                fut = value.iloc[i+1]['Close']
                if (value.iloc[i]['Close']>fut):
                    value.loc[i,'After1']=int(-1)
                elif (value.iloc[i]['Close']<fut):
                    value.loc[i,'After1']=int(1)
                else:
                    value.loc[i,'After1']=int(0)
            for i in range(0, value.shape[0]-5):
                fut = value.iloc[i+5]['Close']
                if (value.iloc[i]['Close']>fut):
                    value.loc[i,'After5']=int(-1)
                elif (value.iloc[i]['Close']<fut):
                    value.loc[i,'After5']=int(1)
                else:
                    value.loc[i,'After5']=int(0)
            for i in range(0, value.shape[0]-20):
                fut = value.iloc[i+20]['Close']
                if (value.iloc[i]['Close']>fut):
                    value.loc[i,'After20']=int(-1)
                elif (value.iloc[i]['Close']<fut):
                    value.loc[i,'After20']=int(1)
                else:
                    value.loc[i,'After20']=int(0)
            value.fillna(0,inplace=True)
            for index, row in value.iterrows():
                dt = datetime.datetime.strptime(row['Date/Time'], '%m/%d/%Y')
                s = Stocks(ticker=row['Ticker'], date=dt, open=row['Open'], high=row['High'], low=row['Low'], close=row['Close'], volume=row['Volume'], after1=row['After1'], after5=row['After5'], after20=row['After20'])
                s.save()
        for key, value in data.items():
            del value['After1']
            del value['After5']
            del value['After20']
        for key, value in data.items():
            value['OBV'] = 0
            for i in range(1, len(value)):
                if value.loc[i, 'Close'] > value.loc[i-1, 'Close']:
                    value.loc[i, 'OBV'] = value.loc[i-1, 'OBV'] + value.loc[i, 'Volume']
                elif value.loc[i, 'Close'] < value.loc[i-1, 'Close']:
                    value.loc[i, 'OBV'] = value.loc[i-1, 'OBV'] - value.loc[i, 'Volume']
                else:
                    value.loc[i, 'OBV'] = value.loc[i-1, 'OBV']
        for key, value in data.items():
            value['ADI'] = acc_dist_index(value['High'], value['Low'], value['Close'], value['Volume'])
            value['EMA3'] = ema_indicator(value['ADI'], n=3)
            value['EMA10'] = ema_indicator(value['ADI'], n=10)
            value['CO'] = value['EMA3'] - value['EMA10']
            value['MACD'] = macd(value['Close'])
            value['Signal'] = ema_indicator(value['MACD'], n=9)
            value['Histogram'] = value['MACD'] - value ['Signal']
            value['Bollinger_High'] = bollinger_hband(value['Close'])
            value['Bollinger_Low'] = bollinger_lband(value['Close'])
            value.fillna(0,inplace=True)
            for index, row in value.iterrows():
                dt = datetime.datetime.strptime(row['Date/Time'], '%m/%d/%Y')
                s = PlainStocks(ticker=row['Ticker'], date=dt, open=row['Open'], high=row['High'], low=row['Low'], close=row['Close'], volume=row['Volume'], obv=row['OBV'], co=row['CO'], macd=row['MACD'], signal=row['Signal'], histogram=row['Histogram'], bollinger_high=row['Bollinger_High'], bollinger_low=row['Bollinger_Low'])
                s.save()
        for key, value in data.items():
            del value['ADI']
            del value['EMA3']
            del value['EMA10']
        for key, value in data.items():
            for i in range(2, len(value)):
                if value.loc[i-1, 'OBV']>value.loc[i, 'OBV']:
                    value.loc[i,'OBV_Comparison'] = -1
                elif value.loc[i-1, 'OBV']==value.loc[i, 'OBV']:
                    value.loc[i,'OBV_Comparison'] = 0
                else:
                    value.loc[i,'OBV_Comparison'] = 1
            for i in range(2, len(value)):
                if value.loc[i, 'OBV']>0:
                    value.loc[i,'OBV_Position'] = 1
                elif value.loc[i, 'OBV']==0:
                    value.loc[i,'OBV_Position'] = 0
                else:
                    value.loc[i,'OBV_Position'] = -1
            for i in range(9, len(value)):
                if value.loc[i, 'CO']>0:
                    value.loc[i,'CO_Position'] = 1
                elif value.loc[i, 'CO']<0:
                    value.loc[i,'CO_Position'] = -1
                else:
                    value.loc[i,'CO_Position'] = 0
            for i in range(9, len(value)):
                if value.loc[i, 'CO']>value.loc[i-1, 'CO']:
                    value.loc[i,'CO_Comparison'] = 1
                elif value.loc[i, 'CO']<value.loc[i-1, 'CO']:
                    value.loc[i,'CO_Comparison'] = -1
                else:
                    value.loc[i,'CO_Comparison'] = 0
            for i in range(33, len(value)):
                if value.loc[i, 'MACD']>0:
                    value.loc[i,'MACD_Position'] = 1
                elif value.loc[i, 'MACD']<0:
                    value.loc[i,'MACD_Position'] = -1
                else:
                    value.loc[i,'MACD_Position'] = 0
            for i in range(33, len(value)):
                if value.loc[i, 'MACD']>value.loc[i-1, 'MACD']:
                    value.loc[i,'MACD_Comparison'] = 1
                elif value.loc[i, 'MACD']<value.loc[i-1, 'MACD']:
                    value.loc[i,'MACD_Comparison'] = -1
                else:
                    value.loc[i,'MACD_Comparison'] = 0
            for i in range(33, len(value)):
                if value.loc[i, 'Signal']>0:
                    value.loc[i,'Signal_Position'] = 1
                elif value.loc[i, 'Signal']<0:
                    value.loc[i,'Signal_Position'] = -1
                else:
                    value.loc[i,'Signal_Position'] = 0
            for i in range(33, len(value)):
                if value.loc[i, 'Signal']>value.loc[i-1, 'Signal']:
                    value.loc[i,'Signal_Comparison'] = 1
                elif value.loc[i, 'Signal']<value.loc[i-1, 'Signal']:
                    value.loc[i,'Signal_Comparison'] = -1
                else:
                    value.loc[i,'Signal_Comparison'] = 0
            for i in range(33, len(value)):
                if value.loc[i, 'Histogram']>0:
                    value.loc[i,'Histogram_Position'] = 1
                elif value.loc[i, 'Histogram']<0:
                    value.loc[i,'Histogram_Position'] = -1
                else:
                    value.loc[i,'Histogram_Position'] = 0
            for i in range(33, len(value)):
                if value.loc[i, 'Histogram']>value.loc[i-1, 'Histogram']:
                    value.loc[i,'Histogram_Comparison'] = 1
                elif value.loc[i, 'Histogram']<value.loc[i-1, 'Histogram']:
                    value.loc[i,'Histogram_Comparison'] = -1
                else:
                    value.loc[i,'Histogram_Comparison'] = 0
            for i in range(19, len(value)):
                if value.loc[i, 'Bollinger_High']<value.loc[i, 'Close']:
                    value.loc[i,'BB_Condition'] = 1
                elif value.loc[i, 'Bollinger_Low']>value.loc[i, 'Close']:
                    value.loc[i,'BB_Condition'] = -1
                else:
                    value.loc[i,'BB_Condition'] = 0
            del value['Open']
            del value['High']
            del value['Low']
            del value['Close']
            del value['Volume']
            del value['OBV']
            del value['CO']
            del value['MACD']
            del value['Signal']
            del value['Histogram']
            del value['Bollinger_High']
            del value['Bollinger_Low']
            value.fillna(0,inplace=True)
            for index, row in value.iterrows():
                dt = datetime.datetime.strptime(row['Date/Time'], '%m/%d/%Y')
                s = GroupedStocks(ticker=row['Ticker'], date=dt, obv_comparison=row['OBV_Comparison'], obv_position=row['OBV_Position'], co_comparison=row['CO_Comparison'], co_position=row['CO_Position'], macd_comparison=row['MACD_Comparison'], macd_position=row['MACD_Position'], signal_comparison=row['Signal_Comparison'], signal_position=row['Signal_Position'], histogram_comparison=row['Histogram_Comparison'], histogram_position=row['Histogram_Position'], bb_condition=row['BB_Condition'])
                s.save()
        for key, value in data.items():
            for i in range(2, len(value)):
                if value.loc[i,'OBV_Comparison'] == -1:
                    value.loc[i,'OBV_Comparison'] = 'Turun'
                elif value.loc[i,'OBV_Comparison'] == 0:
                    value.loc[i,'OBV_Comparison'] = 'Tetap'
                else:
                    value.loc[i,'OBV_Comparison'] = 'Naik'
            for i in range(2, len(value)):
                if value.loc[i,'OBV_Position'] == 1:
                    value.loc[i,'OBV_Position'] = 'Positif'
                elif value.loc[i,'OBV_Position'] == 0:
                    value.loc[i,'OBV_Position'] = 'Nol'
                else:
                    value.loc[i,'OBV_Position'] = 'Negatif'
            for i in range(9, len(value)):
                if value.loc[i,'CO_Position'] == 1:
                    value.loc[i,'CO_Position'] = 'Positif'
                elif value.loc[i,'CO_Position'] == -1:
                    value.loc[i,'CO_Position'] = 'Negatif'
                else:
                    value.loc[i,'CO_Position'] = 'Nol'
            for i in range(9, len(value)):
                if value.loc[i,'CO_Comparison'] == 1:
                    value.loc[i,'CO_Comparison'] = 'Naik'
                elif value.loc[i,'CO_Comparison'] == -1:
                    value.loc[i,'CO_Comparison'] = 'Turun'
                else:
                    value.loc[i,'CO_Comparison'] = 'Tetap'
            for i in range(33, len(value)):
                if value.loc[i,'MACD_Position'] == 1:
                    value.loc[i,'MACD_Position'] = 'Positif'
                elif value.loc[i,'MACD_Position'] == -1:
                    value.loc[i,'MACD_Position'] = 'Negatif'
                else:
                    value.loc[i,'MACD_Position'] = 'Nol'
            for i in range(33, len(value)):
                if value.loc[i,'MACD_Comparison'] == 1:
                    value.loc[i,'MACD_Comparison'] = 'Naik'
                elif value.loc[i,'MACD_Comparison'] == -1:
                    value.loc[i,'MACD_Comparison'] = 'Turun'
                else:
                    value.loc[i,'MACD_Comparison'] = 'Tetap'
            for i in range(33, len(value)):
                if value.loc[i,'Signal_Position'] == 1:
                    value.loc[i,'Signal_Position'] = 'Positif'
                elif value.loc[i,'Signal_Position'] == -1:
                    value.loc[i,'Signal_Position'] = 'Negatif'
                else:
                    value.loc[i,'Signal_Position'] = 'Nol'
            for i in range(33, len(value)):
                if value.loc[i,'Signal_Comparison'] == 1:
                    value.loc[i,'Signal_Comparison'] = 'Naik'
                elif value.loc[i,'Signal_Comparison'] == -1:
                    value.loc[i,'Signal_Comparison'] = 'Turun'
                else:
                    value.loc[i,'Signal_Comparison'] = 'Tetap'
            for i in range(33, len(value)):
                if value.loc[i,'Histogram_Position'] == 1:
                    value.loc[i,'Histogram_Position'] = 'Positif'
                elif value.loc[i,'Histogram_Position'] == -1:
                    value.loc[i,'Histogram_Position'] = 'Negatif'
                else:
                    value.loc[i,'Histogram_Position'] = 'Nol'
            for i in range(33, len(value)):
                if value.loc[i,'Histogram_Comparison'] == 1:
                    value.loc[i,'Histogram_Comparison'] = 'Naik'
                elif value.loc[i,'Histogram_Comparison'] == -1:
                    value.loc[i,'Histogram_Comparison'] = 'Turun'
                else:
                    value.loc[i,'Histogram_Comparison'] = 'Tetap'
            for i in range(19, len(value)):
                if value.loc[i,'BB_Condition'] == 1:
                    value.loc[i,'BB_Condition'] = 'Overbought'
                elif value.loc[i,'BB_Condition'] == -1:
                    value.loc[i,'BB_Condition'] = 'Oversold'
                else:
                    value.loc[i,'BB_Condition'] = 'Normal'
            del value['Ticker']
            value = value.dropna()
            date = value['Date/Time'].values
            del value['Date/Time']
            value = pd.get_dummies(value,prefix=value.columns.tolist())
            value['Date/Time'] = date
            list_of_cols = ['Date/Time','OBV_Comparison_Naik','OBV_Comparison_Tetap','OBV_Comparison_Turun','OBV_Position_Positif','OBV_Position_Negatif','OBV_Position_Nol','CO_Position_Positif','CO_Position_Negatif','CO_Position_Nol','CO_Comparison_Naik','CO_Comparison_Turun','CO_Comparison_Tetap','MACD_Position_Positif','MACD_Position_Negatif','MACD_Position_Nol','MACD_Comparison_Naik','MACD_Comparison_Turun','MACD_Comparison_Tetap','Signal_Position_Positif','Signal_Position_Negatif','Signal_Position_Nol','Signal_Comparison_Naik','Signal_Comparison_Turun','Signal_Comparison_Tetap','Histogram_Position_Positif','Histogram_Position_Negatif','Histogram_Position_Nol','Histogram_Comparison_Naik','Histogram_Comparison_Turun','Histogram_Comparison_Tetap','BB_Condition_Overbought','BB_Condition_Oversold','BB_Condition_Normal']
            for x in set(list_of_cols)-set(value.columns.tolist()):
                value[x]=0
            value = value[list_of_cols]
            value.fillna(0,inplace=True)
            for index, row in value.iterrows():
                dt = datetime.datetime.strptime(row['Date/Time'], '%m/%d/%Y')
                s = OneHotStocks(ticker=key, date=dt, obv_comparison_naik=row['OBV_Comparison_Naik'], obv_comparison_tetap=row['OBV_Comparison_Tetap'], obv_comparison_turun=row['OBV_Comparison_Turun'], obv_position_positif=row['OBV_Position_Positif'], obv_position_negatif=row['OBV_Position_Negatif'], obv_position_nol=row['OBV_Position_Nol'], co_comparison_naik=row['CO_Comparison_Naik'], co_comparison_tetap=row['CO_Comparison_Tetap'], co_comparison_turun=row['CO_Comparison_Turun'], co_position_positif=row['CO_Position_Positif'], co_position_negatif=row['CO_Position_Negatif'], co_position_nol=row['CO_Position_Nol'], macd_comparison_naik=row['MACD_Comparison_Naik'], macd_comparison_tetap=row['MACD_Comparison_Tetap'], macd_comparison_turun=row['MACD_Comparison_Turun'], macd_position_positif=row['MACD_Position_Positif'], macd_position_nol=row['MACD_Position_Nol'], macd_position_negatif=row['MACD_Position_Negatif'], signal_comparison_naik=row['Signal_Comparison_Naik'], signal_comparison_tetap=row['Signal_Comparison_Tetap'], signal_comparison_turun=row['Signal_Comparison_Turun'], signal_position_positif=row['Signal_Position_Positif'], signal_position_nol=row['Signal_Position_Nol'], signal_position_negatif=row['Signal_Position_Negatif'], histogram_comparison_naik=row['Histogram_Comparison_Naik'], histogram_comparison_tetap=row['Histogram_Comparison_Tetap'], histogram_comparison_turun=row['Histogram_Comparison_Turun'], histogram_position_positif=row['Histogram_Position_Positif'], histogram_position_nol=row['Histogram_Position_Nol'], histogram_position_negatif=row['Histogram_Position_Negatif'], bb_condition_overbought=row['BB_Condition_Overbought'], bb_condition_oversold=row['BB_Condition_Oversold'], bb_condition_normal=row['BB_Condition_Normal'])
                s.save()
    return render(request, 'prediksi/home.html')

def update(request):
    context = {
        'idx' : idx
    }
    return render(request, 'prediksi/update.html', context)

def do_update(request):
    ticker = request.POST['ticker']
    date = datetime.datetime.strptime(request.POST['date'], '%Y-%m-%d')
    open = request.POST['open']
    high = request.POST['high']
    low = request.POST['low']
    close = request.POST['close']
    volume = request.POST['volume']
    return redirect(update)

def data_main(request):
    return redirect(data, ticker='ADRO')

def data(request, ticker):
    data = PlainStocks.objects.filter(ticker=ticker)
    context = {
        'data': data,
        'ticker':ticker,
        'idx' :idx
    }
    return render(request, 'prediksi/data.html', context)

def chart_main(request):
    return redirect(chart, ticker='ADRO')

def chart(request, ticker):
    context = {
        'ticker' : ticker,
        'idx' : idx
    }
    return render(request, 'prediksi/chart.html', context)

def result_main(request):
    return redirect(result, ticker='ADRO', d='Aug212015')

def result(request, ticker, d):
    os.chdir(parent)
    date = Stocks.objects.values_list('date',flat=True).filter(ticker=ticker)[32:]
    x = 0
    for i, c in enumerate(d):
        if c.isdigit():
            x = i
            break
    mth = d[:x] if len(d[:x])==3 else d[:x-1]
    day = d[x:x+1] if len(d[x:])==5 else d[x:x+2]
    year = d[-4:]
    d = mth +' '+ day + ' ' + year
    d = datetime.datetime.strptime(d, '%b %d %Y').date()
    decision = Stocks.objects.values('after1','after5','after20').filter(ticker=ticker,date=d)
    plain_data = PlainStocks.objects.values_list('open', 'high', 'low', 'close', 'volume', 'obv', 'co', 'macd', 'signal', 'histogram', 'bollinger_high', 'bollinger_low').filter(ticker=ticker,date=d)
    grouped_data = GroupedStocks.objects.values_list('obv_comparison', 'obv_position', 'co_comparison', 'co_position', 'macd_comparison', 'macd_position', 'signal_comparison', 'signal_position', 'histogram_comparison', 'histogram_position', 'bb_condition').filter(ticker=ticker,date=d)
    onehot_data = OneHotStocks.objects.values_list('obv_comparison_naik','obv_comparison_tetap','obv_comparison_turun','obv_position_positif','obv_position_negatif','obv_position_nol','co_position_positif','co_position_negatif','co_position_nol','co_comparison_naik','co_comparison_turun','co_comparison_tetap','macd_position_positif','macd_position_negatif','macd_position_nol','macd_comparison_naik','macd_comparison_turun','macd_comparison_tetap','signal_position_positif','signal_position_negatif','signal_position_nol','signal_comparison_naik','signal_comparison_turun','signal_comparison_tetap','histogram_position_positif','histogram_position_negatif','histogram_position_nol','histogram_comparison_naik','histogram_comparison_turun','histogram_comparison_tetap','bb_condition_overbought','bb_condition_oversold','bb_condition_normal').filter(ticker=ticker,date=d)
    onehot_data = np.array(list(list(onehot_data)[0])).reshape(1,-1)
    grouped_data = np.array(list(list(grouped_data)[0])).reshape(1,-1)
    plain_data = np.array(list(list(plain_data)[0])).reshape(1,-1)
    confidenceplain1 = 0
    filename = ticker+'1-plain.joblib'
    model = load(filename)
    predictionplain1 = model.predict(plain_data)
    for tree in model.estimators_:
        if predictionplain1+1 == tree.predict(plain_data):
            confidenceplain1 = confidenceplain1 + 1
    confidenceplain5 = 0
    filename = ticker+'5-plain.joblib'
    model = load(filename)
    predictionplain5 = model.predict(plain_data)
    for tree in model.estimators_:
        if predictionplain5+1 == tree.predict(plain_data):
            confidenceplain5 = confidenceplain5 + 1
    confidenceplain20 = 0
    filename = ticker+'20-plain.joblib'
    model = load(filename)
    predictionplain20 = model.predict(plain_data)
    for tree in model.estimators_:
        if predictionplain20+1 == tree.predict(plain_data):
            confidenceplain20 = confidenceplain20 + 1
    confidencegrouped1 = 0
    filename = ticker+'1-grouped.joblib'
    model = load(filename)
    predictiongrouped1 = model.predict(grouped_data)
    for tree in model.estimators_:
        if predictiongrouped1+1 ==tree.predict(grouped_data):
            confidencegrouped1 = confidencegrouped1 + 1
    confidencegrouped5 = 0
    filename = ticker+'5-grouped.joblib'
    model = load(filename)
    predictiongrouped5 = model.predict(grouped_data)
    for tree in model.estimators_:
        if predictiongrouped5+1 == tree.predict(grouped_data):
            confidencegrouped5 = confidencegrouped5 + 1
    confidencegrouped20 = 0
    filename = ticker+'20-grouped.joblib'
    model = load(filename)
    predictiongrouped20 = model.predict(grouped_data)
    for tree in model.estimators_:
        if predictiongrouped20+1 == tree.predict(grouped_data):
            confidencegrouped20 = confidencegrouped20 + 1
    confidenceonehot1 = 0
    filename = ticker+'1-onehot.joblib'
    model = load(filename)
    predictiononehot1 = model.predict(onehot_data)
    for tree in model.estimators_:
        if predictiononehot1+1 == tree.predict(onehot_data):
            confidenceonehot1 = confidenceonehot1 + 1
    confidenceonehot5 = 0
    filename = ticker+'5-onehot.joblib'
    model = load(filename)
    predictiononehot5 = model.predict(onehot_data)
    for tree in model.estimators_:
        if predictiononehot5+1 == tree.predict(onehot_data):
            confidenceonehot5= confidenceonehot5 + 1
    confidenceonehot20 = 0
    filename = ticker+'20-onehot.joblib'
    model = load(filename)
    predictiononehot20 = model.predict(onehot_data)
    for tree in model.estimators_:
        if predictiononehot20+1 == tree.predict(onehot_data):
            confidenceonehot20 = confidenceonehot20 + 1
    context = {
        'd': d,
        'date': date,
        'ticker' : ticker,
        'idx' : idx,
        'after1' : ('Naik' if decision[0]['after1']==1 else ('Turun' if decision[0]['after1']==-1 else 'Tetap')),
        'after5' : ('Naik' if decision[0]['after5']==1 else ('Turun' if decision[0]['after5']==-1 else 'Tetap')),
        'after20' : ('Naik' if decision[0]['after20']==1 else ('Turun' if decision[0]['after20']==-1 else 'Tetap')),
        'predictionplain1' : ('Naik' if predictionplain1==1 else ('Turun' if predictionplain1==-1 else 'Tetap')),
        'confidenceplain1' : confidenceplain1,
        'predictionplain5' : ('Naik' if predictionplain5==1 else ('Turun' if predictionplain5==-1 else 'Tetap')),
        'confidenceplain5' : confidenceplain5,
        'predictionplain20' : ('Naik' if predictionplain20==1 else ('Turun' if predictionplain20==-1 else 'Tetap')),
        'confidenceplain20' : confidenceplain20,
        'predictiongrouped1' : ('Naik' if predictiongrouped1==1 else ('Turun' if predictiongrouped1==-1 else 'Tetap')),
        'confidencegrouped1' : confidencegrouped1,
        'predictiongrouped5' : ('Naik' if predictiongrouped5==1 else ('Turun' if predictiongrouped5==-1 else 'Tetap')),
        'confidencegrouped5' : confidencegrouped5,
        'predictiongrouped20' : ('Naik' if predictiongrouped20==1 else ('Turun' if predictiongrouped20==-1 else 'Tetap')),
        'confidencegrouped20' : confidencegrouped20,
        'predictiononehot1' : ('Naik' if predictiononehot1==1 else ('Turun' if predictiononehot1==-1 else 'Tetap')),
        'confidenceonehot1' : confidenceonehot1,
        'predictiononehot5' : ('Naik' if predictiononehot5==1 else ('Turun' if predictiononehot5==-1 else 'Tetap')),
        'confidenceonehot5' : confidenceonehot5,
        'predictiononehot20' : ('Naik' if predictiononehot20==1 else ('Turun' if predictiononehot20==-1 else 'Tetap')),
        'confidenceonehot20' : confidenceonehot20
    }
    return render(request, 'prediksi/result.html', context)

def train(request):
    os.chdir(parent)
    for ticker in idx:
        max = 0
        stock_train = pd.DataFrame(list(GroupedStocks.objects.values('obv_comparison', 'obv_position', 'co_comparison', 'co_position', 'macd_comparison', 'macd_position', 'signal_comparison', 'signal_position', 'histogram_comparison', 'histogram_position', 'bb_condition').filter(ticker=ticker, date__lte=datetime.date(2018, 7, 1))[32:]))
        stock_test = pd.DataFrame(list(GroupedStocks.objects.values('obv_comparison', 'obv_position', 'co_comparison', 'co_position', 'macd_comparison', 'macd_position', 'signal_comparison', 'signal_position', 'histogram_comparison', 'histogram_position', 'bb_condition').filter(ticker=ticker, date__lte=datetime.date(2019, 1, 1), date__gte=datetime.date(2018, 6, 30))))
        decision_train = pd.DataFrame(list(Stocks.objects.values('after1').filter(ticker=ticker, date__lte=datetime.date(2018, 7, 1))[32:]))
        decision_test = pd.DataFrame(list(Stocks.objects.values('after1').filter(ticker=ticker, date__lte=datetime.date(2019, 1, 1), date__gte=datetime.date(2018, 6, 30))))
        train_data_array = stock_train.values
        train_dec_array = decision_train.values.ravel()
        train_dec_array = train_dec_array.astype('int')
        test_data_array = stock_test.values
        test_dec_array = decision_test.values
        filename = ticker + '1-grouped.joblib'
        for i in range(25):
            clf = RandomForestClassifier(n_estimators=100)
            clf.fit(train_data_array,train_dec_array)
            count_true = 0
            for i in range(len(test_data_array)):
                prediction = clf.predict(test_data_array[i].reshape(1,-1))
                if test_dec_array[i]==prediction:
                    count_true+=1
            acc = count_true/len(test_data_array)
            if acc > max:
                max = acc
                dump(clf,filename)
    for ticker in idx:
        max = 0
        stock_train = pd.DataFrame(list(GroupedStocks.objects.values('obv_comparison', 'obv_position', 'co_comparison', 'co_position', 'macd_comparison', 'macd_position', 'signal_comparison', 'signal_position', 'histogram_comparison', 'histogram_position', 'bb_condition').filter(ticker=ticker, date__lte=datetime.date(2018, 7, 1))[32:]))
        stock_test = pd.DataFrame(list(GroupedStocks.objects.values('obv_comparison', 'obv_position', 'co_comparison', 'co_position', 'macd_comparison', 'macd_position', 'signal_comparison', 'signal_position', 'histogram_comparison', 'histogram_position', 'bb_condition').filter(ticker=ticker, date__lte=datetime.date(2019, 1, 1), date__gte=datetime.date(2018, 6, 30))))
        decision_train = pd.DataFrame(list(Stocks.objects.values('after5').filter(ticker=ticker, date__lte=datetime.date(2018, 7, 1))[32:]))
        decision_test = pd.DataFrame(list(Stocks.objects.values('after5').filter(ticker=ticker, date__lte=datetime.date(2019, 1, 1), date__gte=datetime.date(2018, 6, 30))))
        train_data_array = stock_train.values
        train_dec_array = decision_train.values.ravel()
        train_dec_array = train_dec_array.astype('int')
        test_data_array = stock_test.values
        test_dec_array = decision_test.values
        filename = ticker + '5-grouped.joblib'
        for i in range(25):
            clf = RandomForestClassifier(n_estimators=100)
            clf.fit(train_data_array,train_dec_array)
            count_true = 0
            for i in range(len(test_data_array)):
                prediction = clf.predict(test_data_array[i].reshape(1,-1))
                if test_dec_array[i]==prediction:
                    count_true+=1
            acc = count_true/len(test_data_array)
            if acc > max:
                max = acc
                dump(clf,filename)
    for ticker in idx:
        max = 0
        stock_train = pd.DataFrame(list(GroupedStocks.objects.values('obv_comparison', 'obv_position', 'co_comparison', 'co_position', 'macd_comparison', 'macd_position', 'signal_comparison', 'signal_position', 'histogram_comparison', 'histogram_position', 'bb_condition').filter(ticker=ticker, date__lte=datetime.date(2018, 7, 1))[32:]))
        stock_test = pd.DataFrame(list(GroupedStocks.objects.values('obv_comparison', 'obv_position', 'co_comparison', 'co_position', 'macd_comparison', 'macd_position', 'signal_comparison', 'signal_position', 'histogram_comparison', 'histogram_position', 'bb_condition').filter(ticker=ticker, date__lte=datetime.date(2019, 1, 1), date__gte=datetime.date(2018, 6, 30))))
        decision_train = pd.DataFrame(list(Stocks.objects.values('after20').filter(ticker=ticker, date__lte=datetime.date(2018, 7, 1))[32:]))
        decision_test = pd.DataFrame(list(Stocks.objects.values('after20').filter(ticker=ticker, date__lte=datetime.date(2019, 1, 1), date__gte=datetime.date(2018, 6, 30))))
        train_data_array = stock_train.values
        train_dec_array = decision_train.values.ravel()
        train_dec_array = train_dec_array.astype('int')
        test_data_array = stock_test.values
        test_dec_array = decision_test.values
        filename = ticker + '20-grouped.joblib'
        for i in range(25):
            clf = RandomForestClassifier(n_estimators=100)
            clf.fit(train_data_array,train_dec_array)
            count_true = 0
            for i in range(len(test_data_array)):
                prediction = clf.predict(test_data_array[i].reshape(1,-1))
                if test_dec_array[i]==prediction:
                    count_true+=1
            acc = count_true/len(test_data_array)
            if acc > max:
                max = acc
                dump(clf,filename)
    for ticker in idx:
        max = 0
        stock_train = pd.DataFrame(list(PlainStocks.objects.values('open', 'high', 'low', 'close', 'volume', 'obv', 'co', 'macd', 'signal', 'histogram', 'bollinger_high', 'bollinger_low').filter(ticker=ticker, date__lte=datetime.date(2018, 7, 1))[32:]))
        stock_test = pd.DataFrame(list(PlainStocks.objects.values('open', 'high', 'low', 'close', 'volume', 'obv', 'co', 'macd', 'signal', 'histogram', 'bollinger_high', 'bollinger_low').filter(ticker=ticker, date__lte=datetime.date(2019, 1, 1), date__gte=datetime.date(2018, 6, 30))))
        decision_train = pd.DataFrame(list(Stocks.objects.values('after1').filter(ticker=ticker, date__lte=datetime.date(2018, 7, 1))[32:]))
        decision_test = pd.DataFrame(list(Stocks.objects.values('after1').filter(ticker=ticker, date__lte=datetime.date(2019, 1, 1), date__gte=datetime.date(2018, 6, 30))))
        train_data_array = stock_train.values
        train_dec_array = decision_train.values.ravel()
        train_dec_array = train_dec_array.astype('int')
        test_data_array = stock_test.values
        test_dec_array = decision_test.values
        filename = ticker + '1-plain.joblib'
        for i in range(25):
            clf = RandomForestClassifier(n_estimators=100)
            clf.fit(train_data_array,train_dec_array)
            count_true = 0
            for i in range(len(test_data_array)):
                prediction = clf.predict(test_data_array[i].reshape(1,-1))
                if test_dec_array[i]==prediction:
                    count_true+=1
            acc = count_true/len(test_data_array)
            if acc > max:
                max = acc
                dump(clf,filename)
    for ticker in idx:
        max = 0
        stock_train = pd.DataFrame(list(PlainStocks.objects.values('open', 'high', 'low', 'close', 'volume', 'obv', 'co', 'macd', 'signal', 'histogram', 'bollinger_high', 'bollinger_low').filter(ticker=ticker, date__lte=datetime.date(2018, 7, 1))[32:]))
        stock_test = pd.DataFrame(list(PlainStocks.objects.values('open', 'high', 'low', 'close', 'volume', 'obv', 'co', 'macd', 'signal', 'histogram', 'bollinger_high', 'bollinger_low').filter(ticker=ticker, date__lte=datetime.date(2019, 1, 1), date__gte=datetime.date(2018, 6, 30))))
        decision_train = pd.DataFrame(list(Stocks.objects.values('after5').filter(ticker=ticker, date__lte=datetime.date(2018, 7, 1))[32:]))
        decision_test = pd.DataFrame(list(Stocks.objects.values('after5').filter(ticker=ticker, date__lte=datetime.date(2019, 1, 1), date__gte=datetime.date(2018, 6, 30))))
        train_data_array = stock_train.values
        train_dec_array = decision_train.values.ravel()
        train_dec_array = train_dec_array.astype('int')
        test_data_array = stock_test.values
        test_dec_array = decision_test.values
        filename = ticker + '5-plain.joblib'
        for i in range(25):
            clf = RandomForestClassifier(n_estimators=100)
            clf.fit(train_data_array,train_dec_array)
            count_true = 0
            for i in range(len(test_data_array)):
                prediction = clf.predict(test_data_array[i].reshape(1,-1))
                if test_dec_array[i]==prediction:
                    count_true+=1
            acc = count_true/len(test_data_array)
            if acc > max:
                max = acc
                dump(clf,filename)
    for ticker in idx:
        max = 0
        stock_train = pd.DataFrame(list(PlainStocks.objects.values('open', 'high', 'low', 'close', 'volume', 'obv', 'co', 'macd', 'signal', 'histogram', 'bollinger_high', 'bollinger_low').filter(ticker=ticker, date__lte=datetime.date(2018, 7, 1))[32:]))
        stock_test = pd.DataFrame(list(PlainStocks.objects.values('open', 'high', 'low', 'close', 'volume', 'obv', 'co', 'macd', 'signal', 'histogram', 'bollinger_high', 'bollinger_low').filter(ticker=ticker, date__lte=datetime.date(2019, 1, 1), date__gte=datetime.date(2018, 6, 30))))
        decision_train = pd.DataFrame(list(Stocks.objects.values('after20').filter(ticker=ticker, date__lte=datetime.date(2018, 7, 1))[32:]))
        decision_test = pd.DataFrame(list(Stocks.objects.values('after20').filter(ticker=ticker, date__lte=datetime.date(2019, 1, 1), date__gte=datetime.date(2018, 6, 30))))
        train_data_array = stock_train.values
        train_dec_array = decision_train.values.ravel()
        train_dec_array = train_dec_array.astype('int')
        test_data_array = stock_test.values
        test_dec_array = decision_test.values
        filename = ticker + '20-plain.joblib'
        for i in range(25):
            clf = RandomForestClassifier(n_estimators=100)
            clf.fit(train_data_array,train_dec_array)
            count_true = 0
            for i in range(len(test_data_array)):
                prediction = clf.predict(test_data_array[i].reshape(1,-1))
                if test_dec_array[i]==prediction:
                    count_true+=1
            acc = count_true/len(test_data_array)
            if acc > max:
                max = acc
                dump(clf,filename)
    for ticker in idx:
        max = 0
        stock_train = pd.DataFrame(list(OneHotStocks.objects.values('obv_comparison_naik','obv_comparison_tetap','obv_comparison_turun','obv_position_positif','obv_position_negatif','obv_position_nol','co_position_positif','co_position_negatif','co_position_nol','co_comparison_naik','co_comparison_turun','co_comparison_tetap','macd_position_positif','macd_position_negatif','macd_position_nol','macd_comparison_naik','macd_comparison_turun','macd_comparison_tetap','signal_position_positif','signal_position_negatif','signal_position_nol','signal_comparison_naik','signal_comparison_turun','signal_comparison_tetap','histogram_position_positif','histogram_position_negatif','histogram_position_nol','histogram_comparison_naik','histogram_comparison_turun','histogram_comparison_tetap','bb_condition_overbought','bb_condition_oversold','bb_condition_normal').filter(ticker=ticker, date__lte=datetime.date(2018, 7, 1))[32:]))
        stock_test = pd.DataFrame(list(OneHotStocks.objects.values('obv_comparison_naik','obv_comparison_tetap','obv_comparison_turun','obv_position_positif','obv_position_negatif','obv_position_nol','co_position_positif','co_position_negatif','co_position_nol','co_comparison_naik','co_comparison_turun','co_comparison_tetap','macd_position_positif','macd_position_negatif','macd_position_nol','macd_comparison_naik','macd_comparison_turun','macd_comparison_tetap','signal_position_positif','signal_position_negatif','signal_position_nol','signal_comparison_naik','signal_comparison_turun','signal_comparison_tetap','histogram_position_positif','histogram_position_negatif','histogram_position_nol','histogram_comparison_naik','histogram_comparison_turun','histogram_comparison_tetap','bb_condition_overbought','bb_condition_oversold','bb_condition_normal').filter(ticker=ticker, date__lte=datetime.date(2019, 1, 1), date__gte=datetime.date(2018, 6, 30))))
        decision_train = pd.DataFrame(list(Stocks.objects.values('after1').filter(ticker=ticker, date__lte=datetime.date(2018, 7, 1))[32:]))
        decision_test = pd.DataFrame(list(Stocks.objects.values('after1').filter(ticker=ticker, date__lte=datetime.date(2019, 1, 1), date__gte=datetime.date(2018, 6, 30))))
        train_data_array = stock_train.values
        train_dec_array = decision_train.values.ravel()
        train_dec_array = train_dec_array.astype('int')
        test_data_array = stock_test.values
        test_dec_array = decision_test.values
        filename = ticker + '1-onehot.joblib'
        for i in range(25):
            clf = RandomForestClassifier(n_estimators=100)
            clf.fit(train_data_array,train_dec_array)
            count_true = 0
            for i in range(len(test_data_array)):
                prediction = clf.predict(test_data_array[i].reshape(1,-1))
                if test_dec_array[i]==prediction:
                    count_true+=1
            acc = count_true/len(test_data_array)
            if acc > max:
                max = acc
                dump(clf,filename)
    for ticker in idx:
        max = 0
        stock_train = pd.DataFrame(list(OneHotStocks.objects.values('obv_comparison_naik','obv_comparison_tetap','obv_comparison_turun','obv_position_positif','obv_position_negatif','obv_position_nol','co_position_positif','co_position_negatif','co_position_nol','co_comparison_naik','co_comparison_turun','co_comparison_tetap','macd_position_positif','macd_position_negatif','macd_position_nol','macd_comparison_naik','macd_comparison_turun','macd_comparison_tetap','signal_position_positif','signal_position_negatif','signal_position_nol','signal_comparison_naik','signal_comparison_turun','signal_comparison_tetap','histogram_position_positif','histogram_position_negatif','histogram_position_nol','histogram_comparison_naik','histogram_comparison_turun','histogram_comparison_tetap','bb_condition_overbought','bb_condition_oversold','bb_condition_normal').filter(ticker=ticker, date__lte=datetime.date(2018, 7, 1))[32:]))
        stock_test = pd.DataFrame(list(OneHotStocks.objects.values('obv_comparison_naik','obv_comparison_tetap','obv_comparison_turun','obv_position_positif','obv_position_negatif','obv_position_nol','co_position_positif','co_position_negatif','co_position_nol','co_comparison_naik','co_comparison_turun','co_comparison_tetap','macd_position_positif','macd_position_negatif','macd_position_nol','macd_comparison_naik','macd_comparison_turun','macd_comparison_tetap','signal_position_positif','signal_position_negatif','signal_position_nol','signal_comparison_naik','signal_comparison_turun','signal_comparison_tetap','histogram_position_positif','histogram_position_negatif','histogram_position_nol','histogram_comparison_naik','histogram_comparison_turun','histogram_comparison_tetap','bb_condition_overbought','bb_condition_oversold','bb_condition_normal').filter(ticker=ticker, date__lte=datetime.date(2019, 1, 1), date__gte=datetime.date(2018, 6, 30))))
        decision_train = pd.DataFrame(list(Stocks.objects.values('after5').filter(ticker=ticker, date__lte=datetime.date(2018, 7, 1))[32:]))
        decision_test = pd.DataFrame(list(Stocks.objects.values('after5').filter(ticker=ticker, date__lte=datetime.date(2019, 1, 1), date__gte=datetime.date(2018, 6, 30))))
        train_data_array = stock_train.values
        train_dec_array = decision_train.values.ravel()
        train_dec_array = train_dec_array.astype('int')
        test_data_array = stock_test.values
        test_dec_array = decision_test.values
        filename = ticker + '5-onehot.joblib'
        for i in range(25):
            clf = RandomForestClassifier(n_estimators=100)
            clf.fit(train_data_array,train_dec_array)
            count_true = 0
            for i in range(len(test_data_array)):
                prediction = clf.predict(test_data_array[i].reshape(1,-1))
                if test_dec_array[i]==prediction:
                    count_true+=1
            acc = count_true/len(test_data_array)
            if acc > max:
                max = acc
                dump(clf,filename)
    for ticker in idx:
        max = 0
        stock_train = pd.DataFrame(list(OneHotStocks.objects.values('obv_comparison_naik','obv_comparison_tetap','obv_comparison_turun','obv_position_positif','obv_position_negatif','obv_position_nol','co_position_positif','co_position_negatif','co_position_nol','co_comparison_naik','co_comparison_turun','co_comparison_tetap','macd_position_positif','macd_position_negatif','macd_position_nol','macd_comparison_naik','macd_comparison_turun','macd_comparison_tetap','signal_position_positif','signal_position_negatif','signal_position_nol','signal_comparison_naik','signal_comparison_turun','signal_comparison_tetap','histogram_position_positif','histogram_position_negatif','histogram_position_nol','histogram_comparison_naik','histogram_comparison_turun','histogram_comparison_tetap','bb_condition_overbought','bb_condition_oversold','bb_condition_normal').filter(ticker=ticker, date__lte=datetime.date(2018, 7, 1))[32:]))
        stock_test = pd.DataFrame(list(OneHotStocks.objects.values('obv_comparison_naik','obv_comparison_tetap','obv_comparison_turun','obv_position_positif','obv_position_negatif','obv_position_nol','co_position_positif','co_position_negatif','co_position_nol','co_comparison_naik','co_comparison_turun','co_comparison_tetap','macd_position_positif','macd_position_negatif','macd_position_nol','macd_comparison_naik','macd_comparison_turun','macd_comparison_tetap','signal_position_positif','signal_position_negatif','signal_position_nol','signal_comparison_naik','signal_comparison_turun','signal_comparison_tetap','histogram_position_positif','histogram_position_negatif','histogram_position_nol','histogram_comparison_naik','histogram_comparison_turun','histogram_comparison_tetap','bb_condition_overbought','bb_condition_oversold','bb_condition_normal').filter(ticker=ticker, date__lte=datetime.date(2019, 1, 1), date__gte=datetime.date(2018, 6, 30))))
        decision_train = pd.DataFrame(list(Stocks.objects.values('after20').filter(ticker=ticker, date__lte=datetime.date(2018, 7, 1))[32:]))
        decision_test = pd.DataFrame(list(Stocks.objects.values('after20').filter(ticker=ticker, date__lte=datetime.date(2019, 1, 1), date__gte=datetime.date(2018, 6, 30))))
        train_data_array = stock_train.values
        train_dec_array = decision_train.values.ravel()
        train_dec_array = train_dec_array.astype('int')
        test_data_array = stock_test.values
        test_dec_array = decision_test.values
        filename = ticker + '20-onehot.joblib'
        for i in range(25):
            clf = RandomForestClassifier(n_estimators=100)
            clf.fit(train_data_array,train_dec_array)
            count_true = 0
            for i in range(len(test_data_array)):
                prediction = clf.predict(test_data_array[i].reshape(1,-1))
                if test_dec_array[i]==prediction:
                    count_true+=1
            acc = count_true/len(test_data_array)
            if acc > max:
                max = acc
                dump(clf,filename)
    for ticker in idx:
        stock = pd.DataFrame(list(PlainStocks.objects.values('open', 'high', 'low', 'close', 'volume', 'obv', 'co', 'macd', 'signal', 'histogram', 'bollinger_high', 'bollinger_low').filter(ticker=ticker)[32:]))
        stock_date = pd.DataFrame(list(PlainStocks.objects.values('date').filter(ticker=ticker)[32:]))
        stock = stock.values
        stock_date = stock_date.values
        filename = ticker + '1-plain.joblib'
        print(filename)
        model = load(filename)
        for i in range(len(stock)):
            prediction = model.predict(stock[i].reshape(1,-1))
            s = Stocks.objects.get(ticker=ticker,date=stock_date[i][0])
            s.plain1 = prediction[0]
            s.save()
        filename = ticker + '5-plain.joblib'
        print(filename)
        model = load(filename)
        for i in range(len(stock)):
            prediction = model.predict(stock[i].reshape(1,-1))
            s = Stocks.objects.get(ticker=ticker,date=stock_date[i][0])
            s.plain5 = prediction[0]
            s.save()
        filename = ticker + '20-plain.joblib'
        print(filename)
        model = load(filename)
        for i in range(len(stock)):
            prediction = model.predict(stock[i].reshape(1,-1))
            s = Stocks.objects.get(ticker=ticker,date=stock_date[i][0])
            s.plain20 = prediction[0]
            s.save()
        stock = pd.DataFrame(list(GroupedStocks.objects.values('obv_comparison', 'obv_position', 'co_comparison', 'co_position', 'macd_comparison', 'macd_position', 'signal_comparison', 'signal_position', 'histogram_comparison', 'histogram_position', 'bb_condition').filter(ticker=ticker)[32:]))
        stock_date = pd.DataFrame(list(GroupedStocks.objects.values('date').filter(ticker=ticker)[32:]))
        stock = stock.values
        stock_date = stock_date.values
        filename = ticker + '1-grouped.joblib'
        print(filename)
        model = load(filename)
        for i in range(len(stock)):
            prediction = model.predict(stock[i].reshape(1,-1))
            s = Stocks.objects.get(ticker=ticker,date=stock_date[i][0])
            s.grouped1 = prediction[0]
            s.save()
        filename = ticker + '5-grouped.joblib'
        print(filename)
        model = load(filename)
        for i in range(len(stock)):
            prediction = model.predict(stock[i].reshape(1,-1))
            s = Stocks.objects.get(ticker=ticker,date=stock_date[i][0])
            s.grouped5 = prediction[0]
            s.save()
        filename = ticker + '20-grouped.joblib'
        print(filename)
        model = load(filename)
        for i in range(len(stock)):
            prediction = model.predict(stock[i].reshape(1,-1))
            s = Stocks.objects.get(ticker=ticker,date=stock_date[i][0])
            s.grouped20 = prediction[0]
            s.save()
        stock = pd.DataFrame(list(OneHotStocks.objects.values('obv_comparison_naik','obv_comparison_tetap','obv_comparison_turun','obv_position_positif','obv_position_negatif','obv_position_nol','co_position_positif','co_position_negatif','co_position_nol','co_comparison_naik','co_comparison_turun','co_comparison_tetap','macd_position_positif','macd_position_negatif','macd_position_nol','macd_comparison_naik','macd_comparison_turun','macd_comparison_tetap','signal_position_positif','signal_position_negatif','signal_position_nol','signal_comparison_naik','signal_comparison_turun','signal_comparison_tetap','histogram_position_positif','histogram_position_negatif','histogram_position_nol','histogram_comparison_naik','histogram_comparison_turun','histogram_comparison_tetap','bb_condition_overbought','bb_condition_oversold','bb_condition_normal').filter(ticker=ticker)[32:]))
        stock_date = pd.DataFrame(list(OneHotStocks.objects.values('date').filter(ticker=ticker)[32:]))
        stock = stock.values
        stock_date = stock_date.values
        filename = ticker + '1-onehot.joblib'
        print(filename)
        model = load(filename)
        for i in range(len(stock)):
            prediction = model.predict(stock[i].reshape(1,-1))
            s = Stocks.objects.get(ticker=ticker,date=stock_date[i][0])
            s.onehot1 = prediction[0]
            s.save()
        filename = ticker + '5-onehot.joblib'
        print(filename)
        model = load(filename)
        for i in range(len(stock)):
            prediction = model.predict(stock[i].reshape(1,-1))
            s = Stocks.objects.get(ticker=ticker,date=stock_date[i][0])
            s.onehot5 = prediction[0]
            s.save()
        filename = ticker + '20-onehot.joblib'
        print(filename)
        model = load(filename)
        for i in range(len(stock)):
            prediction = model.predict(stock[i].reshape(1,-1))
            s = Stocks.objects.get(ticker=ticker,date=stock_date[i][0])
            s.onehot20 = prediction[0]
            s.save()
    context = {
        'data' : 'wkwkwk'
    }
    return render(request, 'prediksi/train.html', context)

def tree_main(request):
    return redirect(tree, ticker='ADRO', tipe='plain', jarak='1', number='0')

def tree(request, ticker, tipe, jarak, number):
    os.chdir(parent)
    filename = ticker + jarak + "-" + tipe + ".joblib"
    model = load(filename) 
    tree = model.estimators_[int(number)]
    os.chdir(stat)
    if (tipe=='plain'):
        export_graphviz(tree, feature_names = ['open', 'high', 'low', 'close', 'volume', 'obv', 'co', 'macd', 'signal', 'histogram', 'bollinger_high', 'bollinger_low'], class_names = ['Turun', 'Tetap' , 'Naik'])
    elif(tipe=='grouped'):
        export_graphviz(tree, feature_names = ['obv_comparison', 'obv_position', 'co_comparison', 'co_position', 'macd_comparison', 'macd_position', 'signal_comparison', 'signal_position', 'histogram_comparison', 'histogram_position', 'bb_condition'], class_names = ['Turun', 'Tetap' , 'Naik'])
    else:
        export_graphviz(tree, feature_names = ['obv_comparison_naik','obv_comparison_tetap','obv_comparison_turun','obv_position_positif','obv_position_negatif','obv_position_nol','co_position_positif','co_position_negatif','co_position_nol','co_comparison_naik','co_comparison_turun','co_comparison_tetap','macd_position_positif','macd_position_negatif','macd_position_nol','macd_comparison_naik','macd_comparison_turun','macd_comparison_tetap','signal_position_positif','signal_position_negatif','signal_position_nol','signal_comparison_naik','signal_comparison_turun','signal_comparison_tetap','histogram_position_positif','histogram_position_negatif','histogram_position_nol','histogram_comparison_naik','histogram_comparison_turun','histogram_comparison_tetap','bb_condition_overbought','bb_condition_oversold','bb_condition_normal'], class_names = ['Turun', 'Tetap' , 'Naik'])
    with open("tree.dot") as f:
        dot_graph = f.read()
    g = graphviz.Source(dot_graph)
    g.render(format = 'png')
    context = {
        'ticker' : ticker,
        'idx' : idx,
        'tipe' : tipe,
        't' : ['plain','grouped','onehot'],
        'jarak' : jarak,
        'j' : ['1','5','20'],
        'number' : number,
        'n' : [str(i) for i in range(100)]
    }
    return render(request, 'prediksi/tree.html', context)

def summary(request):
    train = {}
    test = {}
    p1e = []
    p5e = []
    p20e = []
    g1e = []
    g5e = []
    g20e = []
    o1e = []
    o5e = []
    o20e = []
    plain1_train = 0
    plain5_train = 0
    plain20_train = 0
    grouped1_train = 0
    grouped5_train = 0
    grouped20_train = 0
    onehot1_train = 0
    onehot5_train = 0
    onehot20_train = 0
    plain1_test = 0
    plain5_test = 0
    plain20_test = 0
    grouped1_test = 0
    grouped5_test = 0
    grouped20_test = 0
    onehot1_test = 0
    onehot5_test = 0
    onehot20_test = 0
    for ticker in idx:
        print(ticker)
        stock_train = pd.DataFrame(list(Stocks.objects.values('after1','after5','after20','plain1','plain5','plain20','grouped1','grouped5','grouped20','onehot1','onehot5','onehot20').filter(ticker=ticker, date__lte=datetime.date(2018, 7, 1))[32:]))
        stock_test = pd.DataFrame(list(Stocks.objects.values('after1','after5','after20','plain1','plain5','plain20','grouped1','grouped5','grouped20','onehot1','onehot5','onehot20').filter(ticker=ticker, date__lte=datetime.date(2019, 1, 1), date__gte=datetime.date(2018, 6, 30))))
        for i in range(len(stock_train)):
            stock_train.loc[i,'plain1res'] = 1 if stock_train.loc[i, 'after1']==stock_train.loc[i, 'plain1'] else 0
            stock_train.loc[i,'plain5res'] = 1 if stock_train.loc[i, 'after5']==stock_train.loc[i, 'plain5'] else 0
            stock_train.loc[i,'plain20res'] = 1 if stock_train.loc[i, 'after20']==stock_train.loc[i, 'plain20'] else 0
            stock_train.loc[i,'grouped1res'] = 1 if stock_train.loc[i, 'after1']==stock_train.loc[i, 'grouped1'] else 0
            stock_train.loc[i,'grouped5res'] = 1 if stock_train.loc[i, 'after5']==stock_train.loc[i, 'grouped5'] else 0
            stock_train.loc[i,'grouped20res'] = 1 if stock_train.loc[i, 'after20']==stock_train.loc[i, 'grouped20'] else 0
            stock_train.loc[i,'onehot1res'] = 1 if stock_train.loc[i, 'after1']==stock_train.loc[i, 'onehot1'] else 0
            stock_train.loc[i,'onehot5res'] = 1 if stock_train.loc[i, 'after5']==stock_train.loc[i, 'onehot5'] else 0
            stock_train.loc[i,'onehot20res'] = 1 if stock_train.loc[i, 'after20']==stock_train.loc[i, 'onehot20'] else 0
        plain1_train += sum(stock_train['plain1res'])/len(stock_train)
        plain5_train += sum(stock_train['plain5res'])/len(stock_train)
        plain20_train += sum(stock_train['plain20res'])/len(stock_train)
        grouped1_train += sum(stock_train['grouped1res'])/len(stock_train)
        grouped5_train += sum(stock_train['grouped5res'])/len(stock_train)
        grouped20_train += sum(stock_train['grouped20res'])/len(stock_train)
        onehot1_train += sum(stock_train['onehot1res'])/len(stock_train)
        onehot5_train += sum(stock_train['onehot5res'])/len(stock_train)
        onehot20_train += sum(stock_train['onehot20res'])/len(stock_train)
        for i in range(len(stock_test)):
            stock_test.loc[i,'plain1res'] = 1 if stock_test.loc[i, 'after1']==stock_test.loc[i, 'plain1'] else 0
            stock_test.loc[i,'plain5res'] = 1 if stock_test.loc[i, 'after5']==stock_test.loc[i, 'plain5'] else 0
            stock_test.loc[i,'plain20res'] = 1 if stock_test.loc[i, 'after20']==stock_test.loc[i, 'plain20'] else 0
            stock_test.loc[i,'grouped1res'] = 1 if stock_test.loc[i, 'after1']==stock_test.loc[i, 'grouped1'] else 0
            stock_test.loc[i,'grouped5res'] = 1 if stock_test.loc[i, 'after5']==stock_test.loc[i, 'grouped5'] else 0
            stock_test.loc[i,'grouped20res'] = 1 if stock_test.loc[i, 'after20']==stock_test.loc[i, 'grouped20'] else 0
            stock_test.loc[i,'onehot1res'] = 1 if stock_test.loc[i, 'after1']==stock_test.loc[i, 'onehot1'] else 0
            stock_test.loc[i,'onehot5res'] = 1 if stock_test.loc[i, 'after5']==stock_test.loc[i, 'onehot5'] else 0
            stock_test.loc[i,'onehot20res'] = 1 if stock_test.loc[i, 'after20']==stock_test.loc[i, 'onehot20'] else 0
        plain1_test += sum(stock_test['plain1res'])/len(stock_test)
        plain5_test += sum(stock_test['plain5res'])/len(stock_test)
        plain20_test += sum(stock_test['plain20res'])/len(stock_test)
        grouped1_test += sum(stock_test['grouped1res'])/len(stock_test)
        grouped5_test += sum(stock_test['grouped5res'])/len(stock_test)
        grouped20_test += sum(stock_test['grouped20res'])/len(stock_test)
        onehot1_test += sum(stock_test['onehot1res'])/len(stock_test)
        onehot5_test += sum(stock_test['onehot5res'])/len(stock_test)
        onehot20_test += sum(stock_test['onehot20res'])/len(stock_test)
        p1e.append(round(sum(stock_test['plain1res'])/len(stock_test),3))
        p5e.append(round(sum(stock_test['plain5res'])/len(stock_test),3))
        p20e.append(round(sum(stock_test['plain20res'])/len(stock_test),3))
        g1e.append(round(sum(stock_test['grouped1res'])/len(stock_test),3))
        g5e.append(round(sum(stock_test['grouped5res'])/len(stock_test),3))
        g20e.append(round(sum(stock_test['grouped20res'])/len(stock_test),3))
        o1e.append(round(sum(stock_test['onehot1res'])/len(stock_test),3))
        o5e.append(round(sum(stock_test['onehot5res'])/len(stock_test),3))
        o20e.append(round(sum(stock_test['onehot5res'])/len(stock_test),3))
    plain1_train = round(plain1_train/len(idx),3)
    plain5_train = round(plain5_train/len(idx),3)
    plain20_train = round(plain20_train/len(idx),3)
    grouped1_train = round(grouped1_train/len(idx),3)
    grouped5_train = round(grouped5_train/len(idx),3)
    grouped20_train = round(grouped20_train/len(idx),3)
    onehot1_train = round(onehot1_train/len(idx),3)
    onehot5_train = round(onehot5_train/len(idx),3)
    onehot20_train = round(onehot20_test/len(idx),3)
    plain1_test = round(plain1_test/len(idx),3)
    plain5_test = round(plain5_train/len(idx),3)
    plain20_test = round(plain20_test/len(idx),3)
    grouped1_test = round(grouped1_test/len(idx),3)
    grouped5_test = round(grouped5_test/len(idx),3)
    grouped20_test = round(grouped20_test/len(idx),3)
    onehot1_test = round(onehot1_test/len(idx),3)
    onehot5_test = round(onehot5_test/len(idx),3)
    onehot20_test = round(onehot20_test/len(idx),3)
    context = {
        'idx' : idx,
        'plain1_train' : plain1_train,
        'plain1_test' : plain1_test,
        'plain5_train' : plain5_train,
        'plain5_test' : plain5_test,
        'plain20_train' : plain20_train,
        'plain20_test' : plain20_test,
        'grouped1_train' : grouped1_train,
        'grouped1_test' : grouped1_test,
        'grouped5_train' : grouped5_train,
        'grouped5_test' : grouped5_test,
        'grouped20_train' : grouped20_train,
        'grouped20_test' : grouped20_test,
        'onehot1_train' : onehot1_train,
        'onehot1_test' : onehot1_test,
        'onehot5_train' : onehot5_train,
        'onehot5_test' : onehot5_test,
        'onehot20_train' : onehot20_train,
        'onehot20_test' : onehot20_test,
        'p1e' : p1e,
        'p5e' : p5e,
        'p20e' : p20e,
        'g1e' : g1e,
        'g5e' : g5e,
        'g20e' : g20e,
        'o1e' : o1e,
        'o5e' : o5e,
        'o20e' : o20e
    }
    return render(request, 'prediksi/summary.html', context)

def coba(request, ticker):
    data = Stocks.objects.filter(ticker=ticker)
    string = "["
    for row in data:
        string += "["
        string += str((datetime.datetime(row.date.year, row.date.month, row.date.day, 0, 0).timestamp()+86400)*1000)
        string += ","+str(row.open)
        string += ","+str(row.high)
        string += ","+str(row.low)
        string += ","+str(row.close)
        string += ","+str(row.volume)
        string += "],"
    string = string[:-1]
    string += "]"
    context = {
        'string': string,
    }
    return render(request, 'prediksi/coba.html', context)
