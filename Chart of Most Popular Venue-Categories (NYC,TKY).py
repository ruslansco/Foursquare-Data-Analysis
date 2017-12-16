#12/10/2017
#Author: Ruslan Shakirov
#FourSquare Check-Ins EDA
#Will investigate the spatial and temporal aspects of the data provided.

import pandas as pd
import matplotlib as plt
#For plt.figure(figsize)
import matplotlib.pyplot as plt
#Interactive mode for automatic plotting in idle
from matplotlib import interactive
interactive(True)

def drawNYC():
    """
    Function plots 10 most popular Venue Categories in New York.
    """
    #Import dataset of New York from root directory
    NYC = pd.read_csv('dataset_TSMC2014_NYC.csv')

    NYC=NYC[["venueCategory","venueCategoryId"]]
    grouped=NYC.groupby(["venueCategory"]).count()
    grouped=grouped.sort_values('venueCategoryId')
    grouped=grouped[241:251]
    
    #Plot bars of most popular venue categories
    plt.figure(figsize=(16,6))
    plt.style.use('fivethirtyeight')
    plt.bar(grouped.index,grouped["venueCategoryId"])
    plt.title("10 Most Popular Venue Categories \n New York: 2012-2013",fontsize=14,color='black')
    plt.ylabel("Check-ins per Venue Category",fontsize=14)
    plt.show()

def drawTKY():
    """
    Function plots 10 most popular Venue Categories in Tokyo.
    """
    #Import dataset of New York from root directory
    TKY = pd.read_csv('dataset_TSMC2014_TKY.csv')

    TKY=TKY[["venueCategory","venueCategoryId"]]
    grouped2=TKY.groupby(["venueCategory"]).count()
    grouped2=grouped2.sort_values('venueCategoryId')
    grouped2=grouped2[237:247]
    
    #Plot bars of most popular venue categories
    plt.figure(figsize=(16,6))
    plt.style.use('fivethirtyeight')
    plt.bar(grouped2.index,grouped2["venueCategoryId"])
    plt.title("10 Most Popular Venue Categories \n Tokyo: 2012-2013",fontsize=14,color='black')
    plt.ylabel("Check-ins per Venue Category",fontsize=14)
    plt.show()

drawNYC()
drawTKY()
