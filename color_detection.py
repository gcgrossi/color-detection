from skimage import segmentation
from skimage.future import graph
import numpy as np
import os
import cv2
import json
import plotly.express as px
from plotly.subplots import make_subplots
import plotly.graph_objects as go

def make_bar_chart(colormap,topn=10):
    # create a barchart with the top n colors
    # in order of presence
    
    # create a list of 'rgb(r,g,b)' strings
    # 0th component of colormap is (r,g,b)
    rgb = list(map(lambda c: "rgb({},{},{})".format(int(c[0][0]),int(c[0][1]),int(c[0][2])),colormap))
    
    # initialize figure
    fig = go.Figure()
    
    # loop on the first top n elements 
    # of the colormap
    for i,c in enumerate(colormap[:topn]):
        # add a bar chart with y = frequency
        # x = 'r,g,b' 
        # marker color = (r,g,b)
        fig.add_trace(go.Bar(
            x=[rgb[i]],
            y=[c[1]*100],
            name='',
            marker_color=rgb[i]
        ))
        
    # plot figure
    fig.update_layout(showlegend=False, 
                      yaxis=dict(title='[%]'),
                      plot_bgcolor='rgb(255,255,255)')
    fig.show()
    
    return

def make_color_plot(colormap,h=1000,w=1000):
    # create a sliced chart with 
    # all the detected colors
    
    # initialize the x-coordinate
    # and bin-width
    x = bw = 0
    
    # initialize a np array with h,w
    colgrid = np.zeros((h,w,3), dtype='uint8')
    
    # loop on colormap elements
    for cm in colormap:
        # 0th component is the color (r,g,b)
        # 1st component is the frequency
        c,f = cm[0],cm[1]
        # the bin width is proportional
        # to the % frequency of the color
        bw=round(w*f)
        # make a slice h,binwidth
        # with color c
        colgrid[:,x:x+bw] = c
        # increment the x-coordinate
        x+=bw
    
    # plot
    fig = px.imshow(colgrid)
    fig.update_xaxes(showticklabels=False)
    fig.update_yaxes(showticklabels=False)
    fig.show()
    
    return

def get_colormap(color_dict,sort=False):
    # sort the number of pixels for each color
    # if sort bool=True else pass the dictionary
    npixels=sorted(color_dict["npixels"],reverse=True) if sort else color_dict["npixels"]

    # initialize a list for the colormap
    colormap = []

    # calculate total pixels for normalization
    pxtot = sum(npixels)

    for i,npx in enumerate(npixels):
        # get sorted index if bool sort=True
        # else get normal index
        idx = color_dict["npixels"].index(npx) if sort else i
    
        # obtain the corresponding
        # color and frequency
        c = color_dict["color"][idx]
        f = color_dict["npixels"][idx]/pxtot
    
        # push a tuple in the color map with
        # (color, frequency)
        colormap.append((c,f))
        
    return colormap

def color_masks(label_field, image, bg_label=0, bg_color=(0, 0, 0)):
    """Visualise each segment in `label_field` with its mean color in `image`.
    Parameters
    ----------
    label_field : array of int
        A segmentation of an image.
    image : array, shape ``label_field.shape + (3,)``
        A color image of the same spatial shape as `label_field`.
    bg_label : int, optional
        A value in `label_field` to be treated as background.
    bg_color : 3-tuple of int, optional
        The color for the background label
    Returns
    -------
    out : array, same shape and type as `image`
        The output visualization.
    """
    out = np.zeros(label_field.shape + (3,), dtype=image.dtype)
    labels = np.unique(label_field)
    bg = (labels == bg_label)
    if bg.any():
        labels = labels[labels != bg_label]
        mask = (label_field == bg_label).nonzero()
        out[mask] = bg_color
    
    color_info={"color":[],"npixels":[]}
    
    for label in labels:
        mask = (label_field == label).nonzero()
        color = image[mask].mean(axis=0)
        out[mask] = color
        color_info["color"].append(color)
        color_info["npixels"].append(image[mask].shape[0])
    
    color_info["image"] = out
    return color_info

def main():

    # read on input image using cv2
    filename=os.path.join(os.getcwd(),"images","balrog.jpg")
    img_bgr = cv2.imread(filename)

    # convert rgb->bgr (a cv2 speciality)
    img=cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

    # show the input image
    '''
    fig = make_subplots(rows=1, cols=1)
    fig.add_trace(px.imshow(img).data[0],row=1, col=1)
    fig.update_yaxes(showticklabels=False)
    fig.update_xaxes(showticklabels=False)
    fig.show()
    '''

    # get the labels corresponding to the clustered pixels
    labels1 = segmentation.slic(img, compactness=30, n_segments=300)

    # process the labels to obtain the
    # processed image and color information
    color_info = color_masks(labels1, img)
    out1 = color_info["image"]

    # N.B orginal Sci-kit image function
    #out1 = color.label2rgb(labels1, img, kind='avg')

    # create a RAG
    g = graph.rag_mean_color(img, labels1)

    # merge pixels with mean color distance < threshold
    labels2 = graph.cut_threshold(labels1, g, 20)

    # process the labels to obtain the
    # processed image and color information
    color_info2 = color_masks(labels2, img)
    out2 = color_info2["image"]
    out2 = segmentation.mark_boundaries(out2, labels2, (0, 0, 0))

    # N.B orginal Sci-kit image function
    #out2 = color.label2rgb(labels2, img, kind='avg')

    colormap = get_colormap(color_info2,sort=True)
    make_color_plot(colormap)
    make_bar_chart(colormap)

    fig = make_subplots(rows=1, cols=1)
    fig.add_trace(px.imshow(img).data[0],row=1, col=1)
    fig.update_xaxes(showticklabels=False)
    fig.update_yaxes(showticklabels=False)
    fig.show()

    fig = make_subplots(rows=1, cols=1)
    fig.add_trace(px.imshow(out1).data[0],row=1, col=1)
    fig.update_xaxes(showticklabels=False)
    fig.update_yaxes(showticklabels=False)
    fig.show()

    fig = make_subplots(rows=1, cols=1)
    fig.add_trace(px.imshow(out2).data[0],row=1, col=1)
    fig.update_xaxes(showticklabels=False)
    fig.update_yaxes(showticklabels=False)
    fig.show()
    
    return


if __name__ == "__main__":
   main()