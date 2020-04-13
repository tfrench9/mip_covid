#Plotly Dependencies
import plotly
import plotly.graph_objects as go
import plotly.express as px
from plotly.tools import mpl_to_plotly
import plotly.figure_factory as ff

#Dash Dependencies
import dash
import dash_html_components as html
import dash_core_components as dcc
from dash.dependencies import Input, Output
import dash_bootstrap_components as dbc

#Other Dependencies
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import warnings
import json
from skimage.color import rgb2hsv, rgb2lab

#Written Code
import colorNormalization as cn
import extractFeatures as ef

#Global Variable Definations
totalClicks = 0
images = []
preCMImages = []
labels = None
features = None
ldaPoints = None

#Stylesheet and App Configuration
external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']
app = dash.Dash(__name__, external_stylesheets = external_stylesheets)
app.config['suppress_callback_exceptions'] = True
warnings.filterwarnings("ignore")

#Main App Layout
app.layout = html.Div([
    #Title
    html.H3("Georgia Tech ECE4783 Project", style = {'textAlign': 'center'}),
    html.P("Medical Image Processing of Histopathological Images from Cancer Patients", style = {'textAlign': 'center'}),
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
            html.H6('Select You Criteria', style = {'textAlign': 'center', 'padding-top': '20px', 'padding-below': '20px'}),
            dcc.Dropdown(
                id = 'type-dropdown',
                options = [
                    {'label': 'Necrosis', 'value': 'Necrosis'},
                    {'label': 'Stroma', 'value': 'Stroma'},
                    {'label': 'Tumor', 'value': 'Tumor'}
                ],
                placeholder = 'Slelect Image Types to Preprocess',
                multi = True),
            dcc.Dropdown(
                id = 'operations-dropdown',
                options = [
                    {'label': 'Reinhard Color Normalization', 'value': 'N'},
                    {'label': 'Random Crop, Rotate, and Flip', 'value': 'CRF'},
                ],
                placeholder = 'Select Operations',
                multi = True),
            html.P('Select Number of Images Per Class', style = {'textAlign': 'center', 'padding-top': '10px'}),
            dcc.Slider(
                id = 'numImages-slider',
                min = 25, max = 100, step = 1,
                marks={
                    25: '25',
                    100: '100'
                },
                value = 100
            ),
            html.P('Select Image Use', style = {'textAlign': 'center', 'padding-top': '10px'}),
            dcc.Slider(
                id = 'numUses-slider',
                min = 1, max = 10, step = 1,
                marks={
                    1: '1',
                    10: '10'
                },
                value = 5
            ),
            html.Button('Compile Data Set', id = 'compile-button', style = {'width': '40%', 'margin-left': '30%', 'margin-right': '30%'}),
            dcc.Loading(html.Div(id = 'images-content'))
        ],
        style = {
            'width': '80%',
            'padding-left': '10%',
            'padding-right': '10%'
        })
    elif tab == 'p2-tab':
        return html.Div(children = [
            html.H6('Total Images: {}'.format(len(images)), style = {'textAlign': 'center', 'padding-top': '20px', 'padding-below': '20px'}),
            html.Button('Compile Feature Data Set', id = 'features-button', style = {'width': '40%', 'margin-left': '30%', 'margin-right': '30%'}),
            dcc.Loading(html.Div(id = 'features-content')),
            html.Div(id = 'click-data')
        ])
    elif tab == 'p3-tab':
        return html.Div([
            html.H3('Tab content 3')
        ])

#Tab1 callback
@app.callback(Output('images-content', 'children'),
             [Input('compile-button', 'n_clicks'),
              Input('type-dropdown', 'value'),
              Input('operations-dropdown', 'value'),
              Input('numImages-slider', 'value'),
              Input('numUses-slider', 'value')])
def renderTab1Content(clicks, type, operation, numImages, numUse):
    global totalClicks, images, preCMImages, labels, features
    if clicks is not None and clicks > totalClicks:
        if type is not None and operation is not None:
            images, labels = cn.getImages(type, numImages)
            if 'CRF' in operation:
                images, labels = cn.cropRotateFlip(images, labels, numUse)
            preCMImages = images
            features = ef.getColorFeatures(images)
            if 'N' in operation:
                images, lables = cn.colorNormalize(images, labels, numImages)
            showLabels = cn.saveSampleImages(images, labels)
            toAppend = []
            for i in range(25):
                toAppend.append(html.Img(id = 'img{}'.format(i), src = Image.open('DisplayImages/{}.png'.format(i)), style = {'width': '15%', 'padding-left': '2.5%', 'padding-right': '2.5%', 'padding-bottom': '30px'}))
                toAppend.append(dbc.Tooltip(['Origional Image: {}'.format(showLabels[i][0]), html.Br(), 'Crop: {}, Flip: {}, Rotate: {}'.format(showLabels[i][1], showLabels[i][2], showLabels[i][3])], target = 'img{}'.format(i), style = {'textAlign': 'center', 'fontSize': '12px'}))
            totalClicks = clicks
            return html.Div(children = [
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
    global images, labels, features, ldaPoints
    if clicks is not None:
        moreFeatures = ef.getOtherFeatures(images)
        features = np.hstack((features, moreFeatures))
        points, numLabels, variance = ef.performLDA(features, labels)
        if points.shape[1] == 1:
            points = np.hstack((points, np.zeros([points.shape[0], 1])))
        ldaPoints = points
        return [
            html.P('{} Features Extracted Per Image'.format(features.shape[1]), style = {'textAlign': 'center', 'padding-top': '30px'}),
            html.P('{}% of variance explained with frist projected axis and {}% with second axis'.format(round(variance[0] * 100, 2), round(variance[1] * 100, 2)), style = {'textAlign': 'center'}),
            dcc.Graph(
                id = 'lda-graph',
                figure = {
                    'data': [
                        {
                            'x': points[numLabels == 0, 0],
                            'y': points[numLabels == 0, 1],
                            'mode': 'markers',
                            'name': 'Necrosis',
                            'marker': {'size': 8}
                        },
                        {
                            'x': points[numLabels == 1, 0],
                            'y': points[numLabels == 1, 1],
                            'mode': 'markers',
                            'name': 'Stroma',
                            'marker': {'size': 8}
                        },
                        {
                            'x': points[numLabels == 2, 0],
                            'y': points[numLabels == 2, 1],
                            'mode': 'markers',
                            'name': 'Tumor',
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

@app.callback(Output('click-data', 'children'),
            [Input('lda-graph', 'clickData')])
def display_click_data(clickData):
    global images, preCMImages, labels, ldaPoints, features
    if clickData is not None:
        index = np.where(ldaPoints == clickData['points'][0]['x'])[0][0]
        image = images[index]
        preCMImage = preCMImages[index]
        ef.savePicture(preCMImage, 'efOrigionalImage')
        print('image saved')
        rgbFig = ff.create_distplot([list(preCMImage[::4, ::4, 0].flatten()), list(preCMImage[::4, ::4, 1].flatten()), list(preCMImage[::4, ::4, 2].flatten())], ['Red Layer', 'Green Layer', 'Blue Layer'], colors = ['red', 'green', 'blue'], bin_size = 0.02)
        rgbFig.update_layout({'title': 'RBG Histogram', 'xaxis': {'title': 'Color Layer Values', 'range': [0, 1]}, 'yaxis': {'title': 'Count'}, 'plot_bgcolor': 'rgba(0, 0, 0, 0)'})
        print('rbg done')
        preCMImage1 = rgb2hsv(preCMImage)
        hsvFig = ff.create_distplot([list(preCMImage1[::4, ::4, 0].flatten()), list(preCMImage1[::4, ::4, 1].flatten()), list(preCMImage1[::4, ::4, 2].flatten())], ['Hue Layer', 'Saturation Layer', 'Value Layer'], colors = ['red', 'gray', 'black'], bin_size = 0.02)
        hsvFig.update_layout({'title': 'HSV Histogram', 'xaxis': {'title': 'Color Layer Values', 'range': [0, 1]}, 'yaxis': {'title': 'Count'}, 'plot_bgcolor': 'rgba(0, 0, 0, 0)'})
        print('hsv done')
        preCMImage2 = rgb2lab(preCMImage)
        labFig = ff.create_distplot([list(preCMImage2[::4, ::4, 0].flatten()), list(preCMImage2[::4, ::4, 1].flatten()), list(preCMImage2[::4, ::4, 2].flatten())], ['L Layer', 'A Layer', 'B Layer'], colors = ['cyan', 'magenta', 'orange'], bin_size = 0.02)
        labFig.update_layout({'title': 'LAB Histogram', 'xaxis': {'title': 'Color Layer Values', 'range': [0, 1]}, 'yaxis': {'title': 'Count'}, 'plot_bgcolor': 'rgba(0, 0, 0, 0)'})
        print('lab done')
        ef.saveCER(image)
        return [
            html.P('Origional Image Filename: {}'.format(labels[index][0]), style = {'textAlign': 'center', 'padding-top': '30px'}),
            html.P('Crop: {}, Flip: {}, Rotate: {}'.format(labels[index][1], labels[index][2], labels[index][3]), style = {'textAlign': 'center'}),
            html.Img(id = 'lda-image', src = Image.open('DisplayImages/efOrigionalImage.png'), style = {'width': '26%', 'padding-left': '37%', 'padding-right': '37%'}),
            html.H3('Color Features', style = {'textAlign': 'center', 'padding-top': '30px'}),
            html.P('Means and standard deviations of each layer in each color mode were saved as features.', style = {'textAlign': 'center'}),
            dcc.Graph(id = 'rgb-histogram', figure = rgbFig),
            dcc.Graph(id = 'hsv-histogram', figure = hsvFig),
            dcc.Graph(id = 'lab-histogram', figure = labFig),
            html.H3('DCT Features', style = {'textAlign': 'center', 'padding-top': '30px'}),
            html.P('The top 20 frequencies and their magnitudes were saved as features.', style = {'textAlign': 'center'}),
            dcc.Graph(id = 'dct-stem', figure = {'data': [
                {'x': features[index, 24:43], 'y': features[index, 44:63], 'mode': 'markers', 'name': 'Important Frequencies'},
                {'x': features[index, 24:43], 'y': features[index, 44:63], 'type': 'bar', 'name': ''}],
                'layout': {
                    'title': 'Top 20 Frequenceis (Not Including DC Componant)',
                    'xaxis': {'title': 'Index'},
                    'yaxis': {'title': 'Magnitude'},
                }}),
            html.H3('GLCM Features', style = {'textAlign': 'center', 'padding-top': '30px'}),
            html.P('GLCM taken in 8 directions at distances of 1, 3, and 5 were saved as features.', style = {'textAlign': 'center'}),
            html.P('(Contrast, Homogeneity, Energy, Correlation, Entropy)', style = {'textAlign': 'center'}),
            html.P('Distace 1: ({:0.2f}, {:0.2f}, {:0.2f}, {:0.2f}, {:0.2f})'.format(features[index, 64], features[index, 65], features[index, 66], features[index, 67], features[index, 68]), style = {'textAlign': 'center'}),
            html.P('Distace 3: ({:0.2f}, {:0.2f}, {:0.2f}, {:0.2f}, {:0.2f})'.format(features[index, 69], features[index, 70], features[index, 71], features[index, 72], features[index, 73]), style = {'textAlign': 'center'}),
            html.P('Distace 5: ({:0.2f}, {:0.2f}, {:0.2f}, {:0.2f}, {:0.2f})'.format(features[index, 74], features[index, 75], features[index, 76], features[index, 77], features[index, 78]), style = {'textAlign': 'center'}),
            html.H3('Morphological Features', style = {'textAlign': 'center', 'padding-top': '30px'}),
            html.P('Means and standard deviations of edge densitites were saved as features.', style = {'textAlign': 'center'}),
            html.P('(Erosion, Dialation)', style = {'textAlign': 'center'}),
            html.Img(src = Image.open('DisplayImages/erosion.png'), style = {'width': '20%', 'padding-left': '25%', 'padding-right': '5%'}),
            html.Img(src = Image.open('DisplayImages/dialation.png'), style = {'width': '20%', 'padding-left': '5%', 'padding-right': '25%'}),
            html.P('(Canny, Eroded Canny, Dialated Canny)', style = {'textAlign': 'center'}),
            html.Img(src = Image.open('DisplayImages/canny.png'), style = {'width': '20%', 'padding-left': '10%', 'padding-right': '5%'}),
            html.Img(src = Image.open('DisplayImages/erosionEdge.png'), style = {'width': '20%', 'padding-left': '5%', 'padding-right': '5%'}),
            html.Img(src = Image.open('DisplayImages/dialationEdge.png'), style = {'width': '20%', 'padding-left': '5%', 'padding-right': '10%'}),
            html.P('Number of Small Circles: {}'.format(features[index, 103]), style = {'textAlign': 'center'}),
            html.P('Number of Big Circles: {}'.format(features[index, 104]), style = {'textAlign': 'center'})
        ]
        
#Start dis bitch
if __name__ == '__main__':
    app.run_server(debug = True)
