#Plotly Dependencies
import plotly
import plotly.graph_objects as go
import plotly.express as px
from plotly.tools import mpl_to_plotly
import plotly.figure_factory as ff

#Dash Dependencies
import dash
import dash_table
import dash_html_components as html
import dash_core_components as dcc
from dash.dependencies import Input, Output
import dash_bootstrap_components as dbc

#Other Dependencies
import numpy as np
import pandas as pd
from PIL import Image
import matplotlib.pyplot as plt
import warnings
import json
from skimage.color import rgb2hsv, rgb2lab
import random

#Written Code
import colorNormalization as cn
import extractFeatures as ef
import predictionModels as pm

#Global Variable Definations
totalClicks = 0
trainImages = []
testImages = []
trainLabels = []
trainLabels = []
preCNTrainImages = []
lastClicks = 0
trainFeatures = None
testFeatures = None
ldaPoints = None
numLabels = None
points = None
newNumLabels = None
lda = None

#Stylesheet and App Configuration
external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']
app = dash.Dash(__name__, external_stylesheets = external_stylesheets)
app.config['suppress_callback_exceptions'] = True
warnings.filterwarnings("ignore")

#Main App Layout
app.layout = html.Div([
    #Title
    html.H3("Georgia Tech ECE4783 Project", style = {'textAlign': 'center'}),
    html.P("Medical Image Processing of X-Ray Images from COVID-19 Patients", style = {'textAlign': 'center'}),
    #Tabs
    dcc.Tabs(id = "tabs", value = 'p1-tab', children = [
        dcc.Tab(label = 'Image Preprocessing', value = 'p1-tab'),
        dcc.Tab(label='Feature Extraction and Selection', value = 'p2-tab'),
        dcc.Tab(label='Prediction Modeling', value = 'p3-tab'),
    ]),
    #Page Body
    html.Div(id = 'tabs-content')
])

#Tabbing callback
@app.callback(Output('tabs-content', 'children'),
             [Input('tabs', 'value')])
def renderTabContent(tab):
    if tab == 'p1-tab':
        return html.Div(children = [
            html.H6('Select Your Criteria', style = {'textAlign': 'center', 'padding-top': '20px', 'padding-below': '20px'}),
            dcc.Dropdown(
                id = 'type-dropdown',
                options = [
                    {'label': 'Healthy', 'value': 'No'},
                    {'label': 'Pneumothorax', 'value': 'Pn'},
                    {'label': 'COVID-19', 'value': 'CO'}
                ],
                placeholder = 'Slelect Image Types to Preprocess',
                value = ['No', 'Pn', 'CO'],
                multi = True),
            dcc.Dropdown(
                id = 'operations-dropdown',
                options = [
                    {'label': 'Color Normalization', 'value': 'N'},
                    {'label': 'Sharpen/Blur', 'value': 'SB'},
                    {'label': 'Gaussian Noise', 'value': 'GN'},
                ],
                placeholder = 'Select Operations',
                value = ['N', 'SB', 'GN'],
                multi = True),
            html.P('Select Percent of Images Per Class', style = {'textAlign': 'center', 'padding-top': '30px'}),
            dcc.Slider(
                id = 'numImages-slider',
                min = 25, max = 100, step = 1,
                marks = {
                    25: '25%',
                    100: '100%'
                },
                value = 100
            ),
            html.P('Select Max Sharpen to Max Blur Range (If Applicable)', style = {'textAlign': 'center', 'padding-top': '10px'}),
            dcc.RangeSlider(
                id = 'sb-slider',
                min = -3, max = 3, step = 0.1,
                marks = {
                    -3: '3 Blur',
                    0: 'No Operation',
                    3: '3 Sharpen'
                },
                value = [-1, 1]
            ),
            html.P('Select Gaussian Noise Strength (If Applicable)', style = {'textAlign': 'center', 'padding-top': '30px'}),
            dcc.Slider(
                id = 'gn-slider',
                min = 0, max = 10, step = 0.1,
                marks = {
                    0: '0',
                    10: '10'
                },
                value = 3
            ),
            html.P('Select Train/Test Split Percentage', style = {'textAlign': 'center', 'padding-top': '10px'}),
            dcc.Slider(
                id = 'tts-slider',
                min = 0, max = 100, step = 1,
                marks = {
                    0: '0%',
                    100: '100%'
                },
                value = 80
            ),
            html.Button('Compile Data Set', id = 'compile-button', style = {'width': '40%', 'margin-top': '30px', 'margin-left': '30%', 'margin-right': '30%'}),
            dcc.Loading(html.Div(id = 'images-content'))
        ],
        style = {
            'width': '80%',
            'padding-left': '10%',
            'padding-right': '10%'
        })
    elif tab == 'p2-tab':
        return html.Div(children = [
            html.H6('Total Training Images: {}'.format(len(trainImages)), style = {'textAlign': 'center', 'padding-top': '20px', 'padding-below': '20px'}),
            html.Button('Compile Feature Data Set', id = 'features-button', style = {'width': '40%', 'margin-left': '30%', 'margin-right': '30%'}),
            dcc.Loading(html.Div(id = 'features-content')),
            html.Div(id = 'click-data')
        ])
    elif tab == 'p3-tab':
        return html.Div([
            html.H6('Test Basic Prediction Algorithms', style = {'textAlign': 'center', 'padding-top': '20px', 'padding-below': '20px'}),
            html.Button('See Testing Data Projected on LDA Feature Space', id = 'prediction-button', style = {'width': '40%', 'margin-left': '30%', 'margin-right': '30%'}),
            html.Div(id = 'prediction-content')
        ])

#Tab1 callback
@app.callback(Output('images-content', 'children'),
             [Input('compile-button', 'n_clicks'),
              Input('type-dropdown', 'value'),
              Input('operations-dropdown', 'value'),
              Input('numImages-slider', 'value'),
              Input('sb-slider', 'value'),
              Input('gn-slider', 'value'),
              Input('tts-slider', 'value')])
def renderTab1Content(clicks, type, operation, percentImages, sb, noise, tts):
    global totalClicks, trainImages, testImages, trainLabels, testLabels, preCNTrainImages, trainFeatures, testFeatures
    if clicks is not None and clicks > totalClicks:
        if type is not None and operation is not None:
            #Read in images from the file
            trainImages, testImages, trainLabels, testLabels = cn.getImages(type, percentImages / 100, tts / 100)
            #Copy image array and extract pre-normilization color features
            preCNTrainImages = trainImages
            #trainFeatures = ef.getColorFeatures(trainImages)
            #testFeatures = ef.getColorFeatures(testImages)
            #Apply the selected operations
            if 'SB' in operation:
                trainImages, trainLabels = cn.blurSharpen(trainImages, trainLabels, sb[0], sb[1])
            if 'GN' in operation:
                trainImages, trainLabels = cn.gaussianNoise(trainImages, trainLabels, noise)
            if 'N' in operation:
                trainImages = cn.colorNormalize(trainImages)
                testImages = cn.colorNormalize(testImages)
            #Compile a random set of images to display
            showLabels = cn.saveSampleImages(trainImages, trainLabels)
            toAppend = []
            for i in range(25):
                toAppend.append(html.Img(id = 'img{}'.format(i), src = Image.open('DisplayImages/{}.png'.format(i)), style = {'width': '15%', 'padding-left': '2.5%', 'padding-right': '2.5%', 'padding-bottom': '30px'}))
                toAppend.append(dbc.Tooltip(['Diagnosis: {}, Patient ID: {}'.format(showLabels[i][0], showLabels[i][1]), html.Br(), 'L/R Lung {}, Blur/Sharpen: {}, Noise: {}'.format(showLabels[i][3], showLabels[i][4], showLabels[i][5])], target = 'img{}'.format(i), style = {'textAlign': 'center', 'fontSize': '12px'}))
            totalClicks = clicks
            return html.Div(children = [
                    html.H6('{} Training Images, {} Testing Images'.format(len(trainImages), len(testImages)), style = {'textAlign': 'center', 'padding-bottom': '30px'}),
                    html.H6('25 Sample Images', style = {'textAlign': 'center', 'padding-bottom': '30px'}),
                    html.Div(children = toAppend)
                ],
                style = {'padding-top': '30px'}
            )
        else:
            return [html.P('Make Sure All Fields Have Valid Entries', style = {'textAlign': 'center', 'padding-top': '150px'})]
    else:
        return [html.P('Awaiting Valid User Input', style = {'textAlign': 'center', 'padding-top': '150px'})]

#Tab2 callback
@app.callback(Output('features-content', 'children'),
             [Input('features-button', 'n_clicks')])
def renderTab2Content(clicks):
    global trainImages, testImages, trainLabels, testLabels, trainFeatures, testFeatures, ldaPoints, numLabels, lda
    if clicks is not None:
        trainFeatures = ef.getOtherFeatures(trainImages)
        trainFeaturesDF = pd.DataFrame(data = trainFeatures)
        trainFeaturesDF.insert(0, "Class", [x[0] for x in trainLabels])
        trainFeaturesDF.insert(1, "PID", [x[1] for x in trainLabels])
        trainFeaturesDF.insert(2, "L/R Lung", [x[3] for x in trainLabels])
        trainFeaturesDF.to_csv("trainFeatures.csv")
        #trainFeatures = np.hstack((trainFeatures, h1))
        testFeatures = ef.getOtherFeatures(testImages)
        #testFeatures = np.hstack((testFeatures, h2))
        testFeaturesDF = pd.DataFrame(data = testFeatures)
        testFeaturesDF.insert(0, "Class", [x[0] for x in testLabels])
        testFeaturesDF.insert(1, "PID", [x[1] for x in testLabels])
        testFeaturesDF.insert(2, "L/R Lung", [x[3] for x in testLabels])
        testFeaturesDF.to_csv("testFeatures.csv")
        points, numLabels, variance, lda = ef.performLDA(trainFeatures, trainLabels)
        if points.shape[1] == 1:
            points = np.hstack((points, np.zeros([points.shape[0], 1])))
        ldaPoints = points
        return [
            html.P('{} Features Extracted Per Image'.format(trainFeatures.shape[1]), style = {'textAlign': 'center', 'padding-top': '30px'}),
            html.P('{}% of variance explained with frist projected axis and {}% with second axis'.format(round(variance[0] * 100, 2), round(variance[1] * 100, 2)), style = {'textAlign': 'center'}),
            dcc.Graph(
                id = 'lda-graph',
                figure = {
                    'data': [
                        {
                            'x': points[numLabels == 0, 0],
                            'y': points[numLabels == 0, 1],
                            'mode': 'markers',
                            'name': 'Healthy',
                            'marker': {'size': 8}
                        },
                        {
                            'x': points[numLabels == 1, 0],
                            'y': points[numLabels == 1, 1],
                            'mode': 'markers',
                            'name': 'Pneumothorax',
                            'marker': {'size': 8}
                        },
                        {
                            'x': points[numLabels == 2, 0],
                            'y': points[numLabels == 2, 1],
                            'mode': 'markers',
                            'name': 'COVID-19',
                            'marker': {'size': 8}
                        }
                    ],
                    'layout': {
                        'title': 'LDA Projection of Image Features',
                        'xaxis': {'title': 'Principal Axis 1'},
                        'yaxis': {'title': 'Principal Axis 2'},
                        'clickmode': 'event+select'
                    }
                }
            )
        ]

#Show T2 Point Click Specs
@app.callback(Output('click-data', 'children'),
            [Input('lda-graph', 'clickData')])
def displayClickData(clickData):
    global trainImages, testImages, trainLabels, testLabels, trainFeatures, testFeatures, ldaPoints, numLabels, lda, preCNTrainImages
    if clickData is not None:
        index = np.where(ldaPoints == clickData['points'][0]['x'])[0][0]
        trainImage = trainImages[index]
        preCNTrainImage = preCNTrainImages[index]
        ef.savePicture(preCNTrainImage, 'efOrigionalImage')
        colorFig = ff.create_distplot([list(preCNTrainImage[::4, ::4, 0].flatten())], ['Intensity'], colors = ['black'], bin_size = 0.02)
        colorFig.update_layout({'title': 'Intensity Histogram', 'xaxis': {'title': 'Intensity Value', 'range': [0, 1]}, 'yaxis': {'title': 'Count'}, 'plot_bgcolor': 'rgba(0, 0, 0, 0)'})
        return [
            html.P('Origional Image Type: {}, Patient ID: {}'.format(trainLabels[index][0], trainLabels[index][1]), style = {'textAlign': 'center', 'padding-top': '30px'}),
            html.P('L/R Lung: {}, Blur/Sharpen: {}, Noise: {}'.format(trainLabels[index][3], trainLabels[index][4], trainLabels[index][5]), style = {'textAlign': 'center'}),
            html.Img(id = 'lda-image', src = Image.open('DisplayImages/efOrigionalImage.png'), style = {'width': '26%', 'padding-left': '37%', 'padding-right': '37%'}),
            html.H3('Color Features', style = {'textAlign': 'center', 'padding-top': '30px'}),
            html.P('Mean and Standard Deviation of the Intensity wer Saved as Features.', style = {'textAlign': 'center'}),
            dcc.Graph(id = 'rgb-histogram', figure = colorFig),
            #html.H3('DCT Features', style = {'textAlign': 'center', 'padding-top': '30px'}),
            #html.P('The top 20 frequencies and their magnitudes were saved as features.', style = {'textAlign': 'center'}),
            #dcc.Graph(id = 'dct-stem', figure = {'data': [
            #    {'x': features[index, 24:43], 'y': features[index, 44:63], 'mode': 'markers', 'name': 'Important Frequencies'},
            #    {'x': features[index, 24:43], 'y': features[index, 44:63], 'type': 'bar', 'name': ''}],
            #    'layout': {
            #        'title': 'Top 20 Frequenceis (Not Including DC Componant)',
            #        'xaxis': {'title': 'Index'},
            #        'yaxis': {'title': 'Magnitude'},
            #    }}),
            #html.H3('GLCM Features', style = {'textAlign': 'center', 'padding-top': '30px'}),
            #html.P('GLCM taken in 8 directions at distances of 1, 3, and 5 were saved as features.', style = {'textAlign': 'center'}),
            #html.P('(Contrast, Homogeneity, Energy, Correlation, Entropy)', style = {'textAlign': 'center'}),
            #html.P('Distace 1: ({:0.2f}, {:0.2f}, {:0.2f}, {:0.2f}, {:0.2f})'.format(features[index, 64], features[index, 65], features[index, 66], features[index, 67], features[index, 68]), style = {'textAlign': 'center'}),
            #html.P('Distace 3: ({:0.2f}, {:0.2f}, {:0.2f}, {:0.2f}, {:0.2f})'.format(features[index, 69], features[index, 70], features[index, 71], features[index, 72], features[index, 73]), style = {'textAlign': 'center'}),
            #html.P('Distace 5: ({:0.2f}, {:0.2f}, {:0.2f}, {:0.2f}, {:0.2f})'.format(features[index, 74], features[index, 75], features[index, 76], features[index, 77], features[index, 78]), style = {'textAlign': 'center'}),
            #html.H3('Morphological Features', style = {'textAlign': 'center', 'padding-top': '30px'}),
            #html.P('Means and standard deviations of edge densitites were saved as features.', style = {'textAlign': 'center'}),
            #html.P('(Erosion, Dialation)', style = {'textAlign': 'center'}),
            #html.Img(src = Image.open('DisplayImages/erosion.png'), style = {'width': '20%', 'padding-left': '25%', 'padding-right': '5%'}),
            #html.Img(src = Image.open('DisplayImages/dialation.png'), style = {'width': '20%', 'padding-left': '5%', 'padding-right': '25%'}),
            #html.P('(Canny, Eroded Canny, Dialated Canny)', style = {'textAlign': 'center'}),
            #html.Img(src = Image.open('DisplayImages/canny.png'), style = {'width': '20%', 'padding-left': '10%', 'padding-right': '5%'}),
            #html.Img(src = Image.open('DisplayImages/erosionEdge.png'), style = {'width': '20%', 'padding-left': '5%', 'padding-right': '5%'}),
            #html.Img(src = Image.open('DisplayImages/dialationEdge.png'), style = {'width': '20%', 'padding-left': '5%', 'padding-right': '10%'}),
            #html.P('Number of Small Circles: {}'.format(features[index, 103]), style = {'textAlign': 'center'}),
            #html.P('Number of Big Circles: {}'.format(features[index, 104]), style = {'textAlign': 'center'})
        ]

#Tab3 callback
@app.callback(Output('prediction-content', 'children'),
             [Input('prediction-button', 'n_clicks')])
def renderTab3Content(clicks):
    global trainImages, testImages, trainLabels, testLabels, trainFeatures, testFeatures, ldaPoints, numLabels, lda, points, newNumLabels
    if clicks is not None:
        points, newNumLabels = ef.performProjectionLDA(testFeatures, testLabels, lda)
        if points.shape[1] == 1:
            points = np.hstack((points, np.zeros([points.shape[0], 1])))
        return [
            dcc.Graph(
                id = 'new-lda-graph',
                figure = {
                    'data': [
                        {
                            'x': ldaPoints[numLabels == 0, 0],
                            'y': ldaPoints[numLabels == 0, 1],
                            'mode': 'markers',
                            'opacity': 0.5,
                            'name': 'Healthy',
                            'marker': {'size': 8}
                        },
                        {
                            'x': ldaPoints[numLabels == 1, 0],
                            'y': ldaPoints[numLabels == 1, 1],
                            'mode': 'markers',
                            'opacity': 0.5,
                            'name': 'Pneumothorax',
                            'marker': {'size': 8}
                        },
                        {
                            'x': ldaPoints[numLabels == 2, 0],
                            'y': ldaPoints[numLabels == 2, 1],
                            'mode': 'markers',
                            'opacity': 0.5,
                            'name': 'COVID-19',
                            'marker': {'size': 8}
                        },
                        {
                            'x': points[newNumLabels == 0, 0],
                            'y': points[newNumLabels == 0, 1],
                            'mode': 'markers',
                            'name': 'Testing Healthy',
                            'marker': {'size': 8, 'symbol': 'x', 'color': '#1f77b4'}
                        },
                        {
                            'x': points[newNumLabels == 1, 0],
                            'y': points[newNumLabels == 1, 1],
                            'mode': 'markers',
                            'name': 'Testing Pneumothorax',
                            'marker': {'size': 8, 'symbol': 'x', 'color': '#ff7f0e'}
                        },
                        {
                            'x': points[newNumLabels == 2, 0],
                            'y': points[newNumLabels == 2, 1],
                            'mode': 'markers',
                            'name': 'Testing COVID-19',
                            'marker': {'size': 8, 'symbol': 'x', 'color': '#2ca02c'}
                        },

                    ],
                    'layout': {
                        'title': 'Testing Data Projected onto Trained LDA Feature Space',
                        'xaxis': {'title': 'Principal Axis 1'},
                        'yaxis': {'title': 'Principal Axis 2'}
                    }
                }
            ),
            html.Div(children = [
                dcc.Dropdown(
                    id = 'alg-dropdown',
                    options = [
                        {'label': 'Naive Bayes', 'value': 'NB'},
                        {'label': 'K-Means (Nearest Centroid)', 'value': 'KM'},
                        {'label': 'K Nearest Neighbors', 'value': 'KNN'},
                        {'label': 'Support Vector Machine', 'value': 'SVM'},
                    ],
                    placeholder = 'Slelect Algorithms to Comapre',
                    multi = True),
                html.Button('View Parameter Selections', id = 'parameters-button', style = {'width': '40%', 'margin-left': '30%', 'margin-right': '30%'})],
                style = {
                    'width': '80%',
                    'padding-bottom': '30px',
                    'padding-left': '10%',
                    'padding-right': '10%'}
            ),
            html.Div(id = 'parameter-selection')
        ]

#Parameter Callback
@app.callback(Output('parameter-selection', 'children'),
             [Input('parameters-button', 'n_clicks'),
              Input('alg-dropdown', 'value')])
def displayParameterContent(clicks, algs):
    if clicks is not None and algs is not None:
        toReturn = [html.H6('Please Select Parameters', style = {'textAlign': 'center', 'padding-top': '30px'})]
        if 'NB' in algs:
            toReturn.append(
                html.Div(children = [
                    html.P('Naive Bayes: No Parameter Selection Required', style = {'textAlign': 'center', 'padding-top': '30px'})
                ], style = {'width': '50%', 'padding-left': '25%', 'padding-right': '25%'})
            )
        if 'KM' in algs:
            toReturn.append(
                html.Div(children = [
                    html.P('K-Means: Select Number of Clusters (k)', style = {'textAlign': 'center', 'padding-top': '30px'}),
                    dcc.Slider(
                        id = 'KM-slider',
                        min = 1, max = 5, step = 1,
                        marks = {
                            1: '1',
                            5: '5'
                        },
                        value = 3
                    ),
                ], style = {'width': '50%', 'padding-left': '25%', 'padding-right': '25%'})
            )
        if 'KNN' in algs:
            toReturn.append(
                html.Div(children = [
                    html.P('K Nearest Neighbors: Select Number of Neighboring Points (k)', style = {'textAlign': 'center', 'padding-top': '30px'}),
                    dcc.Slider(
                        id = 'KNN-slider',
                        min = 1, max = 20, step = 1,
                        marks = {
                            1: '1',
                            20: '20'
                        },
                        value = 5
                    ),
                ], style = {'width': '50%', 'padding-left': '25%', 'padding-right': '25%'})
            )
        if 'SVM' in algs:
            toReturn.append(
                html.Div(children = [
                    html.P('Support Vector Machine: Select the Kernal', style = {'textAlign': 'center', 'padding-top': '30px'}),
                    dcc.Dropdown(
                        id = 'SVM-dropdown',
                        options = [
                            {'label': 'Linear', 'value': 'linear'},
                            {'label': 'Radial Basis Function', 'value': 'rbf'},
                            {'label': 'Polynomial', 'value': 'poly'},
                            {'label': 'Sigmoid', 'value': 'sigmoid'},
                        ],
                        placeholder = 'Slelect Desired Kernel',
                        multi = False),
                    html.P('Select the Regularization Parameter (C)', style = {'textAlign': 'center'}),
                    dcc.Slider(
                        id = 'C-slider',
                        min = 1, max = 5, step = 0.1,
                        marks = {
                            1: '1',
                            5: '5'
                        },
                        value = 1),
                ], style = {'width': '50%', 'padding-left': '25%', 'padding-right': '25%'})
            )
        toReturn.append(html.Button('Run All', id = 'run-button', style = {'width': '40%', 'margin-left': '30%', 'margin-right': '30%'}))
        toReturn.append(dcc.Loading(html.Div(id = 'results')))
        return toReturn
    else:
        return [html.P('Awaiting Valid User Input', style = {'textAlign': 'center', 'padding-top': '30px'})]

#Parameter Callback
@app.callback(Output('results', 'children'),
             [Input('run-button', 'n_clicks'),
              Input('alg-dropdown', 'value'),
              Input('KM-slider', 'value'),
              Input('KNN-slider', 'value'),
              Input('SVM-dropdown', 'value'),
              Input('C-slider', 'value')])
def runResults(clicks, algs, kmK, knnK, svmKernel, svmC):
    global trainImages, testImages, trainLabels, testLabels, trainFeatures, testFeatures, ldaPoints, numLabels, lda, points, newNumLabels, lastClicks
    if clicks is not None and clicks > lastClicks:
        lastClicks = clicks
        data = []
        if 'NB' in algs:
            r = pm.naiveBays(ldaPoints, numLabels, points, newNumLabels)
            data.append(pm.formatResults('Naive Bayes', r))
        if 'KM' in algs:
            r = pm.kMeans(ldaPoints, numLabels, points, newNumLabels, kmK)
            data.append(pm.formatResults('K-Means (k = {})'.format(kmK), r))
        if 'KNN' in algs:
            r = pm.kNN(ldaPoints, numLabels, points, newNumLabels, knnK)
            data.append(pm.formatResults('K Nearest Neighbors (k = {})'.format(knnK), r))
        if 'SVM' in algs:
            r = pm.svm(ldaPoints, numLabels, points, newNumLabels, svmKernel, svmC)
            data.append(pm.formatResults('SVM ({} kernel, c = {})'.format(svmKernel, svmC), r))
        cols = ['Classifier', 'Overall Accuracy', 'Healthy Properly Labeled', 'Healthy Labeled as Pneumothorax', 'Healthy Labeled as COVID-19', 'Pneumothorax Properly Labeled', 'Pneumothorax Labeled as Healthy', 'Pneumothorax Labeled as COVID-19', 'COVID-19 Properly Labeled', 'COVID-19 Labeled as Healthy', 'COVID-19 Labeled as Pneumothorax']
        df = pd.DataFrame(data = data, columns = cols)
        return [html.Div(children = [
            dash_table.DataTable(
                id = 'results-table',
                columns = [{"name": i, "id": i} for i in df.columns],
                data = df.to_dict('records'),
                style_table = {'overflowX': 'scroll'}
            )
        ], style = {'width': '80%', 'padding-top': '30px', 'padding-bottom': '30px', 'padding-left': '10%','padding-right': '10%'})]
    else:
        pass


#Start dis bitch
if __name__ == '__main__':
    app.run_server(debug = True)
