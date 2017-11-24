#11/16/2017
#Author: Ruslan Shakirov
#Data Visualization of Foursquare check-ins - New York & Tokyo.

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

#Interactive mode for automatic plotting in idle
from matplotlib import interactive
interactive(True)

def drawNYC():
    """
    Function plots all Foursquare check-ins of New York.
    Period: 2012-2013.
    """
    #Import dataset of New York from root directory
    NYC = pd.read_csv('dataset_TSMC2014_NYC.csv')

    #Get GPS Coordinates
    x1 = NYC['longitude'].values
    y1 = NYC['latitude'].values

    #Plot scatter, where x-axis is latitude and y-axis is longitude
    plt.figure(figsize=(12,12))
    ax1 = plt.subplot()
    ax1.scatter(x1,y1,s=8, alpha=0.1, c='#CF000F')

    #Set title and labels for axises
    ax1.set_title("Check-ins of New York \n Period: 2012-2013",fontsize=14,color='black')
    ax1.set_xlabel('Longitude')
    ax1.set_ylabel('Latitude')

def drawTKY():
    """
    Function plots all Foursquare check-ins of Tokyo.
    Period: 2012-2013.
    """
    #Import dataset of Tokyo from root directory
    TKY = pd.read_csv('dataset_TSMC2014_TKY.csv')
    
    #Get GPS Coordinates
    x2 = TKY['longitude'].values
    y2 = TKY['latitude'].values

    #Plot scatter, where x-axis is latitude and y-axis is longitude
    plt.figure(figsize=(12,12))
    ax2 = plt.subplot()
    ax2.scatter(x2,y2,s=8, alpha=0.1, c='#CF000F')
    
    #Set title and labels for axises
    ax2.set_title("Check-ins of Tokyo \n Period: 2012-2013",fontsize=14,color='black')
    ax2.set_xlabel('Longitude')
    ax2.set_ylabel('Latitude')

drawNYC()
drawTKY()
