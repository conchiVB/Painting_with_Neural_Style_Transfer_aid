#!/usr/bin/env python
# coding: utf-8

# In[ ]:


#!pip install streamlit
#!pip3 install --upgrade tensorflow
#!pip3 install --upgrade keras


# In[3]:


import streamlit as st
import os
import asyncio
from PIL import Image

import sys
sys.path.append('/content/drive/MyDrive/TFM/')
import Paleta_picture
import Neural_style_transfer

from skimage.segmentation import slic
from skimage.color import label2rgb
import matplotlib.pyplot as plt
import numpy as np

path='/content/drive/MyDrive/TFM/tmp/'

def get_or_create_eventloop():
    try:
        return asyncio.get_event_loop()
    except RuntimeError as ex:
        if "There is no current event loop in thread" in str(ex):
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            return asyncio.get_event_loop()

def save_uploadedfile(uploadedfile, path,resize):
    
    filename=uploadedfile.name
    filename_new="tmp_image." + 'jpg' #filename.split(".")[-1]    
    with open(os.path.join(path,filename_new),"wb") as f:
        f.write(uploadedfile.getbuffer())
        #st.success(\"Saved File:{} to ./data/tmp/\".format(uploadedfile.name))
        
    img = Image.open(path+filename_new)       
    if img.size[0] != resize:
        factor = (resize/float(img.size[0]))
        wsize = int((float(img.size[1])*float(factor)))
        img = img.resize((resize,wsize),  Image.Resampling.LANCZOS) 
        img.save(path + 'tmp_resize_image.jpg')
    
    return filename_new  
    

def color_segmentation():
    
    nameInFile="tmp_style_transfer_resize.jpg"
    original_image = plt.imread(path + nameInFile,np.uint8)
    
    lst_Nsegments=[50,100,200,500,1000,2000,4000,8000] #1, 1500,2000, 4000
    #lst_segments =[]
    lst_img_segments=[]
    
    for n, Nsegm in enumerate(lst_Nsegments):
        segments_k = slic(original_image, n_segments=Nsegm, start_label=1,compactness = 10)
        #lst_segments.append(segments_k)
        img_segments_k=label2rgb(segments_k, original_image, kind="avg",bg_label=0)
        lst_img_segments.append(img_segments_k/255.)
        # Put segments on top of original image to compare
        #label2rgb(segments_k, original_image, kind="avg",bg_label=0)
        
    return lst_img_segments
    
        
def run_process():
    
    #delete previous output
    #os.remove(path + "tmp_image.jpg") 
    #os.remove(path + "tmp_palette_colours.png") 
    #os.remove(path + "tmp_resize_image.jpg") 
    #os.remove(path + "tmp_style_transfer_resize.jpg") 
    #os.remove(path + "tmp_style_transfer.jpg") 
    #os.remove(path + "tmp_df_colours.csv")
    
    col1, col2= st.columns(2) # divide the display in 2 cols
    #col1, col2= st.columns([10,10]) # divide the display in 2 cols
    st.sidebar.write("## Pick a painting :gear:")
    uploaded_file = st.sidebar.file_uploader("Upload an image", type=["png", "jpg", "jpeg"])
    #uploaded_file = st.file_uploader(label='Pick a painting to test')
    if uploaded_file is not None:
        image_data = uploaded_file.getvalue()        
        #st.image(image_data)
        name_file = uploaded_file.name 
        
        filename_new=save_uploadedfile(uploaded_file,path,500)
        
        if filename_new:
            
            Neural_style_transfer.style_transfer()
            df_colours=Paleta_picture.exact_color(path,'tmp_style_transfer.jpg', 500, 10, 2.5,500)
            df_colours.to_csv(path + 'tmp_df_colours.csv',index=False)
            
    
            
            #df_colours.to_csv(path + 'tmp_df_colours.csv',index=False)
            #image_resize= Image.open(path + 'tmp_resize_image.jpg')
            
            with col1:
                st.header("Original image")
                st.image(path + 'tmp_resize_image.jpg',width=300) #path + 'tmp_resize_image.jpg'
                st.write("___")
                
                st.header("Styled image")
                st.image(path + 'tmp_style_transfer.jpg')
                st.write("___")
                
                
            with col2: 
                st.markdown("# Style image - palette", unsafe_allow_html=False)
                st.write("___")
                st.write("## Palette")
                st.image(path + 'tmp_palette_colours.png')
                st.write("___")
                
                steps=color_segmentation()
                st.write("## Stages")
                for k in range(len(steps)):
                    st.image(steps[k])
                    st.write("___")
                
                
                
          
    

        #col1.image(image_resize, 'Original image')#, use_column_width=True
        #image_colors= Image.open('./data/tmp/tmp_palette_colours.png')
        #col1.image(image_colors, caption='Palette of colours') 
        
        #path_in = uploaded_file.name
        #print(path_in)
        #%run -m Paleta_picture
        

        
def main():
    
    #get_or_create_eventloop()
    st.set_page_config(layout="wide", page_title="Painting suggester")
    #st.title('Painting suggester')
    run_process()
    
    
    
    

if __name__ == '__main__':
    main()


# In[ ]:





# In[ ]:




