import numpy as np
import matplotlib.pyplot as plt
#import matplotlib.colors as mcolors
import pandas as pd
#import datetime

from sklearn.linear_model import  LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import  PolynomialFeatures


plt.style.use('fivethirtyeight')

def get_worldData():
    timeSeries_caseData=pd.read_csv("dataset/world/time_series_covid19_confirmed_global.csv")
    timeSeries_recoverData=pd.read_csv("dataset/world/time_series_covid19_recovered_global.csv")
    timeSeries_deathData=pd.read_csv("dataset/world/time_series_covid19_deaths_global.csv")
    return timeSeries_caseData,timeSeries_recoverData,timeSeries_deathData

def get_calcTotal(data):
    col = data.keys()
    sumData=[]
    for i in col:
        sum=data[i].sum()
        sumData.append(sum)
    return sumData

def check_daily_increment(data):
    daily_data=[]
    for i in range(len(data)):
        if i==0:
            daily_data.append(data[i])
        else:
            daily_data.append(data[i]-data[i-1])
    return daily_data

def plotData(D1,D2,D3,D4,D5,type):
    a="Daily "
    if type==1:
        a="Total "

    plt.plot(D1)
    plt.plot(D2)
    plt.plot(D3)
    plt.plot(D4)
    plt.plot(D5)
    plt.legend([a+"Confirmed Case",a+"Recovered",a+"Death",a+"Closed Case",a+"Active Case"])
    plt.show()


def changeNP(data):
    data=np.array([data[i] for i in range(len(data))]).reshape(-1,1)
    return data


def get_preprocess_worldData():
    timeSeries_caseData,timeSeries_recoverData,timeSeries_deathData= get_worldData()

    col = timeSeries_caseData.keys()

    caseData=timeSeries_caseData.loc[:,col[4]:col[-1]]
    recoverData=timeSeries_recoverData.loc[:,col[4]:col[-1]]
    deathData=timeSeries_deathData.loc[:,col[4]:col[-1]]

    dates = caseData.keys()
        #print(cols)
        #print(caseData)

    confirmed_Case= np.asarray(get_calcTotal(caseData))
    total_Recovered= np.asarray(get_calcTotal(recoverData))
    total_Death= np.asarray(get_calcTotal(deathData))

        #print(total_Recovered)
        #print(total_Death)

    closed_Case=np.asarray(np.array([total_Recovered[i]+total_Death[i] for i in range(len(dates))]))
    total_Active=np.asarray(np.array([confirmed_Case[i]-closed_Case[i] for i in range(len(dates))]))

        #print(closed_Case)

        #daily Data
    new_Case= np.asarray(check_daily_increment(confirmed_Case))
    new_Recovered= np.asarray(check_daily_increment(total_Recovered))
    new_Death= np.asarray(check_daily_increment(total_Death))
    new_ClosedCase= np.asarray(check_daily_increment(closed_Case))
    new_Active= np.asarray(check_daily_increment(total_Active))

    plotData(confirmed_Case, total_Recovered, total_Death, closed_Case, total_Active, 1)
    plotData(np.log(confirmed_Case), np.log(total_Recovered), np.log(total_Death), np.log(closed_Case), np.log(total_Active), 1)
    plotData(new_Case, new_Recovered, new_Death, new_ClosedCase, new_Active, 0)
    plotData(np.log(new_Case), np.log(new_Recovered), np.log(new_Death), np.log(new_ClosedCase), np.log(new_Active), 0)

    days=np.array([i for i in range(len(dates))]).reshape(-1, 1)
    confirmed_Case=changeNP(confirmed_Case)
    total_Active=changeNP(total_Active)

    return days,confirmed_Case,total_Recovered,total_Death,closed_Case,total_Active,new_Case,new_Recovered,new_Death,new_ClosedCase,new_Active


def calcCost(input,output):
    from sklearn.metrics import mean_absolute_error,mean_squared_error,r2_score
    print("mean absolute error =" ,mean_absolute_error(input,output))
    #print("mean squared error= ",mean_squared_error(input,output))
    #print("r2 score = ",r2_score(input,output))

def get_IndiaData():
    data=pd.read_csv("dataset/india/case_time_series.csv")
    return data

def get_process_IndiaData():
    indiaData=get_IndiaData()


    #totalCase=np.asarray(indiaData["Total Confirmed"].replace(0,1))

    totalCase = np.asarray(indiaData["Total Confirmed"].replace(0,1))
    totalRecovered=np.asarray(indiaData["Total Recovered"].replace(0,1))
    totalDeath=np.asarray(indiaData["Total Deceased"].replace(0,1))
    totalClosed=np.array([totalRecovered[i]+totalDeath[i] for i in range(len(totalCase))]).reshape(-1,1)
    totalActive=np.array([totalCase[i]-totalClosed[i] for i in range(len(totalCase))]).reshape(-1,1)
    #print(np.asarray(totalDeath))

    id=np.array([i for i in range(len(totalCase))]).reshape(-1,1)

    dailyCase=np.asarray(indiaData["Daily Confirmed"].replace(0,1))
    dailyRecovered=np.asarray(indiaData["Daily Recovered"].replace(0,1))
    dailyDeath=np.asarray(indiaData["Daily Deceased"].replace(0,1))
    dailyClosed=np.array([dailyRecovered[i]+dailyDeath[i] for i in range(len(dailyCase))]).reshape(-1,1)
    dailyActive=np.array([dailyCase[i]-dailyClosed[i] for i in range(len(dailyCase))]).reshape((-1,1))

    plotData(totalCase,totalRecovered,totalDeath,totalClosed,totalActive,1)
    plotData(np.log(totalCase),np.log(totalRecovered),np.log(totalDeath),np.log(totalClosed),np.log(totalActive),1)

    plotData(dailyCase,dailyRecovered,dailyDeath,dailyClosed,dailyActive,0)
    plotData(np.log(dailyCase),np.log(dailyRecovered),np.log(dailyDeath),np.log(dailyClosed),np.log(dailyActive),0)
    return id,totalCase,totalRecovered,totalDeath,totalClosed,totalActive,dailyCase,dailyRecovered,dailyDeath,dailyClosed,dailyActive


def fitintoLR(x,y1,y2):
    ############################################
    #fitting


    days, X_test, data, y_test = train_test_split(x, y1, test_size=0.08,shuffle=False)
    print(len(X_test))
    poly=PolynomialFeatures(degree=3)
    x_poly=poly.fit_transform(days)
    poly.fit(x_poly,data)
    newCase=LinearRegression()
    newCase.fit(x_poly,data)
    train_pred=newCase.predict(poly.fit_transform(days))
    #plt.plot(days,train_pred)
    #plt.show()


    #################################
    #prediction
    test = np.array([i for i in range(250)]).reshape(-1, 1)
    #test=X_test
    NCgraph=newCase.predict(poly.fit_transform(test))
    #plt.plot(test,np.exp(NCgraph))
    #plt.show()
    test_pred=newCase.predict(poly.fit_transform(X_test))
    print("Confirm Case")
    print("Train")
    calcCost(data,train_pred)
    print("Test")
    calcCost(y_test,test_pred)
    #NCgraph=np.exp(NCgraph)

    ##############################
    # calculation
    e=1
    sum=0
    CCgraph=[]
    a=1;
    for i in range(len(NCgraph)):
        if np.exp(NCgraph[i])<e and a==1 and i>100:
            print("last new case= ",i)
            a=0

        sum=sum+np.exp(NCgraph[i])
        if i==0:
            CCgraph.append(np.exp(NCgraph[i]))
        else:
            CCgraph.append(sum)

    CCgraph=np.asarray(CCgraph)
    plt.plot(test,CCgraph)
    plt.show()
    #print("Confirmed case")
    #print(np.exp(NCgraph[100]))
    #print(CCgraph[100])
    #print(CCgraph[364])
    #calcCost()


    #y1=np.array([y1[i] for i in range(len(y1))]).reshape(-1,1)

    #print("Active")
    #x1=np.concatenate([x,np.log(y1)],axis=1)



    #################################
    # fitting
    X_test,days, y_test,data = train_test_split(x, y2, test_size=0.95, shuffle=False)
    # calcCost()
    poly1=PolynomialFeatures(degree=2)
    x_poly=poly1.fit_transform(days)
    poly1.fit(x_poly,data)
    active=LinearRegression()
    active.fit(x_poly,data)
    train_pred = active.predict(poly1.fit_transform(days))
    #plt.plot([days[i][0] for i in range(len(days))], np.exp(train_pred))
    #plt.show()

    ###############
    # prediction
    test_pred = active.predict(poly1.fit_transform(X_test))
    print("Closed case")
    print("Train")
    calcCost(data, train_pred)
    print("Test")
    calcCost(y_test, test_pred)

    #test1=np.concatenate([test,NCgraph],axis=1)
    ACgraph = active.predict(poly1.fit_transform(test))
    #plt.plot([test[i][0] for i in range(len(test))], np.exp(ACgraph))
    #plt.show()


    ###############################
    # Calculation
    sum = 0
    RCgraph = []
    a = 1;
    for i in range(len(ACgraph)):
        sum = sum + np.exp(ACgraph[i])
        if i == 0:
            RCgraph.append(np.exp(ACgraph[i]))
        else:
            RCgraph.append(sum)

    RCgraph = np.asarray(RCgraph)
    plt.plot(test, RCgraph)
    plt.show()
    print("Confirmed case")
    #print(np.exp(ACgraph[100]))
    #print(RCgraph[100])
    #print(RCgraph[364])


    ####################
    # plot
    DCgraph=np.array([CCgraph[i]-RCgraph[i] for i in range(len(CCgraph))]).reshape(-1,1)
    plt.plot(test,CCgraph)
    plt.plot(test,RCgraph)
    plt.plot(test,DCgraph)
    plt.legend([" Predicted Confirmed Case", " Predicted Closed Case", "Predicted Active Case"])
    plt.show()


    for i in range(len(DCgraph)):
        if DCgraph[i]<0:
            print("last active= ",i)
            break


    plt.plot(test,NCgraph)
    plt.plot(test,ACgraph)
    plt.legend([" Daily NewCase Model", " Daily ClosedCase Model"])
    plt.show()

    print("total case = ",np.ceil(np.amax(CCgraph)))
    print("pick active case = ",np.ceil(np.amax(DCgraph)))
    print("pick active case appear at = ",np.argmax(DCgraph))


def fitintoLR1(x,y1,x2,y2):
    ############################################
    # fitting
    X_test,days, y_test,data = train_test_split(x, y1, test_size=0.64,shuffle=False)
    print(len(X_test))
    poly=PolynomialFeatures(degree=2)
    x_poly=poly.fit_transform(days)
    poly.fit(x_poly,data)
    newCase=LinearRegression()
    newCase.fit(x_poly,data)
    train_pred=newCase.predict(poly.fit_transform(days))
    #plt.plot(days,train_pred)
    # plt.show()
    e=20

    #######################################
    # prediction
    test = np.array([i for i in range(250)]).reshape(-1, 1)
    #test=X_test
    NCgraph=newCase.predict(poly.fit_transform(test))
    #plt.plot(test,np.exp(NCgraph))
    #plt.show()
    test_pred=newCase.predict(poly.fit_transform(X_test))
    print("Confirmed Case")
    print("Train")
    calcCost(data,train_pred)
    print("Test")
    calcCost(y_test,test_pred)
    #NCgraph=np.exp(NCgraph)

    ##############################
    # calculation

    sum=0
    CCgraph=[]
    a=1;
    for i in range(len(NCgraph)):
        if np.exp(NCgraph[i])<e and a==1 and i>100:
            #print("last new case= ",i)
            a=0

        sum=sum+np.exp(NCgraph[i])
        if i==0:
            CCgraph.append(np.exp(NCgraph[i]))
        else:
            CCgraph.append(sum)

    CCgraph=np.asarray(CCgraph)
    plt.plot(test,CCgraph)
    plt.show()
    #print("Confirmed case")
    #print(np.exp(NCgraph[100]))
    #print(CCgraph[100])
    #print(CCgraph[364])
    #calcCost()


    #y1=np.array([y1[i] for i in range(len(y1))]).reshape(-1,1)

    #print("Active")
    #x1=np.concatenate([x,np.log(y1)],axis=1)


    ################################
    # fitting
    X_test,days, y_test,data = train_test_split(x2, y2, test_size=0.52, shuffle=False)#91
    # calcCost()
    #print(len(days))
    poly1=PolynomialFeatures(degree=2)#3
    x_poly=poly1.fit_transform(days)
    poly1.fit(x_poly,data)
    active=LinearRegression()
    active.fit(x_poly,data)
    train_pred = active.predict(poly1.fit_transform(days))
    #plt.plot([days[i][0] for i in range(len(days))], np.exp(train_pred))
    #plt.show()

    #test1=np.concatenate([test,NCgraph],axis=1)


    ###################################
    # prediction
    test_pred = active.predict(poly1.fit_transform(X_test))
    print("Closed case")
    print("Train")
    calcCost(data, train_pred)
    print("Test")
    calcCost(y_test, test_pred)
    ###################
    ACgraph = active.predict(poly1.fit_transform(test))
    #plt.plot([test[i][0] for i in range(len(test))], np.exp(ACgraph))
    #plt.show()

    ########################################
    # calculation
    sum = 0
    RCgraph = []
    a = 1;
    for i in range(len(ACgraph)):

        sum = sum + np.exp(ACgraph[i])
        if i == 0:
            RCgraph.append(np.exp(ACgraph[i]))
        else:
            RCgraph.append(sum)

    RCgraph = np.asarray(RCgraph)
    plt.plot(test, RCgraph)
    plt.show()
    print("Confirmed case")
    #print(np.exp(ACgraph[100]))
    #print(RCgraph[100])
    #print(RCgraph[364])


    ###############################
    # plot
    DCgraph=np.array([CCgraph[i]-RCgraph[i] for i in range(len(CCgraph))]).reshape(-1,1)
    plt.plot(test,CCgraph)
    plt.plot(test,RCgraph)
    plt.plot(test,DCgraph)
    plt.legend([" Predicted Confirmed Case", " Predicted Closed Case", "Predicted Active Case"])
    plt.show()

    for i in range(len(DCgraph)):

        if DCgraph[i]<0:
            print("last active= ",i+1)
            break


    plt.plot(test,NCgraph)
    plt.plot(test,ACgraph)
    plt.legend([" Daily NewCase Model", " Daily ClosedCase Model"])
    plt.show()

    print("total case = ",np.ceil(np.amax(CCgraph)))
    print("pick active case = ",np.ceil(np.amax(DCgraph)))
    print("pick active case appear at = ",np.argmax(DCgraph))



def main():
    days,confirmCase,recovered,death,closedCase,active,newCase,newRecovered,newDeath,newClosedCase,newActive=get_preprocess_worldData()



    fitintoLR(days,np.log(newCase),np.log(newClosedCase))


    #fitintoANN(days,active)
    #fitintosvr(days,active)

    #fitinto(days,newCase)

    #fitin(days,confirmCase)

    id,totalCase,totalRecovered,totalDeath,totalClosed,totalActive,dailyCase,dailyRecovered,dailyDeath,dailyClosed,dailyActive=get_process_IndiaData()


    fitintoLR1(id,np.log(dailyCase),id,np.log(dailyClosed))


if __name__=="__main__":
    main()
