# Painting with Neural Style Transfer as aid
___


Generate a painting-like image using a neural style transfer model to transform the input image into a brush painting style.



### Objective

Given an input image, perform neural style transfer to transform it into a brush stroke painting. The generated image is then analyzed by breaking down its colors to display a visual palette. Finally, a first suggestion for the painting is provided using color segmentation to create a series of images


### Style image

The style image used for the transformation is the next:

<img src="./06_transfer_style/style/paint1.jpg" width="200" height="250" />


### Model

The VGG19 network, which is widely used in style transfer algorithms, is employed as the model in this case. This pre-trained network is utilized to extract the content and style representations from the input images. The extracted features are then employed to generate a new image that merges the content of one image with the style of another image."


### Visualitation

A streamlit app has been developed for the process. The user inputs an image and the program returns the transformed image, a color palette, and a series of images showing the painting process.


### Installation guide

*	git clone https://github.com/conchiVB/Painting_with_Neural_Style_Transfer_aid.git
*	Upload repository to Google Drive
*	Access Google Colab with an account: https://www.datahack.es/google-colab-for-data-science/

#### Developers - Training the model 

From Google drive, open notebook  4_Neural_style_transfer.ipynb. 

To replicate the model results with the same settings, run the complete notebook.

For any changes on the settings:
*	Content and style images; look for the next lines in the code:
    -	CONTENT_IMAGE_URL = path_content + 'puppy.jpg'
    -	STYLE_IMAGE_URL = path_style + 'paint1.jpg'
*	Style and content layers
    -	Style-layer weights: weight1,weight2,weight3,weight4,weight5= 0.1, 0.3, 0.4, 0.1, 0.1
    -	STYLE_LAYERS = [('block1_conv2', weight1),('block2_conv2', weight2),('block3_conv3', weight3),('block4_conv3', weight4), ('block5_conv1', weight5) ]
    -	Content layer in the call to the model; LOAD_VGG19('BLOCK5_CONV1')  
*	Remaining parameters:
    -	LEARNING_RATE = 0.03 #0.05
    -	ALPHA = 1 
    -	BETA =  0.3 
    -	EPOCHS = 2000 
    -	ADD_NOISE = True
    -	NOISE_RANGE = 0.2

#### Client - Visualitation

From Google colab, open streamlit_tfm.py and run the complete code.

The last line !npx localtunnel --port 8501 opens an url, follow the link.

### More information 

[TFM paint Style Transfer](TFM_paint_Style_Transfer.pdf)
