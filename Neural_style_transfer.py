import tensorflow as tf
import numpy as np
import cv2

import pandas as pd
import matplotlib.pyplot as plt

#TF_ENABLE_ONEDNN_OPTS=0
# In[25]:

#COLAB
#from google.colab.patches import cv2_imshow
#from google.colab import drive
#drive.mount('/content/drive')


# In[26]:
#COLAB
path='/content/drive/MyDrive/TFM/tmp/'

path_main='/content/drive/MyDrive/TFM/06_transfer_style/'
path_style = path_main + 'style/'
#path content is now the streamlit image loaded
path_content = path_main + 'content/'




# In[18]:

def crop_center(image):
    """Returns a cropped square image."""
    shape = image.shape
    new_shape = min(shape[0], shape[1])
    offset_y = max(shape[0] - shape[1], 0) // 2
    offset_x = max(shape[1] - shape[0], 0) // 2
    crop_img = image[offset_y:offset_y+new_shape, offset_x:offset_x+new_shape]  
    return crop_img


def load_image(url, width, height, color_mode):
    image = cv2.imread(url)
    image = crop_center(image)
    image = cv2.resize(image, (height, width))
    
    image = cv2.cvtColor(image, color_mode)
    
    image = np.reshape(image, ((1,) + image.shape))
    image = tf.Variable(tf.image.convert_image_dtype(image, tf.float32)) 

    return image
    
STYLE_IMAGE_URL = path_style + 'paint1.jpg'
CONTENT_IMAGE_URL = path + 'tmp_resize_image.jpg'


IMG_HEIGHT, IMG_WIDTH=400,400
COLOR_CHANNELS = 3
weight1,weight2,weight3,weight4,weight5= 0.1, 0.3, 0.4, 0.1, 0.1


# Define the model, where the last layers of VGG19 have been removed. Those layers (dense + softmax activation) are especific of classification models.



# In[24]:
def add_noise(image, noise_range):
    noise = tf.random.uniform(image.shape, -noise_range, noise_range)

    image = tf.add(image, noise)
    image = tf.clip_by_value(image, clip_value_min=0.0, clip_value_max=1.0)
    image = tf.Variable(tf.image.convert_image_dtype(image, tf.float32))

    return image


#  load the model VGG19
def load_VGG19(contentlayer='block5_conv1'):

    STYLE_LAYERS = [
        ('block1_conv2', weight1),
        ('block2_conv2', weight2),
        ('block3_conv3', weight3),
        ('block4_conv3', weight4),
        ('block5_conv1', weight5)
    ]

    model = tf.keras.applications.VGG19(
        include_top = False,
        input_shape = (IMG_HEIGHT, IMG_WIDTH, COLOR_CHANNELS),
        weights = 'imagenet'
        )

    CONTENT_LAYER = [(contentlayer, 1)]  #default -- block5_conv2 

    model.trainable = False

    return model, STYLE_LAYERS, CONTENT_LAYER

model, STYLE_LAYERS, CONTENT_LAYER = load_VGG19('block5_conv1') 
outputs = [model.get_layer(layer[0]).output for layer in STYLE_LAYERS + CONTENT_LAYER]
model = tf.keras.Model([model.input], outputs)

# ### **Content cost function:**
def compute_content_cost(content_output, generated_output):
    a_C = content_output[-1]
    a_G = generated_output[-1]

    m, n_H, n_W, n_C = a_G.get_shape().as_list()

    a_C_unrolled = tf.reshape(a_C, shape=[m, n_H * n_W, n_C])
    a_G_unrolled = tf.reshape(a_G, shape=[m, n_H * n_W, n_C])

    J_content =  tf.reduce_sum(tf.square(a_C_unrolled - a_G_unrolled)) / (4.0 * n_H * n_W * n_C)
    
    return J_content


# 
# ### **Single layer style cost function:**
def gram_matrix(A):
    GA = tf.matmul(A, tf.transpose(A))    
    return GA

def compute_layer_style_cost(a_S, a_G):
    m, n_H, n_W, n_C = a_G.get_shape().as_list()
    
    a_S = tf.transpose(tf.reshape(a_S, shape=[-1, n_C]))
    a_G = tf.transpose(tf.reshape(a_G, shape=[-1, n_C]))
    
    GS = gram_matrix(a_S)
    GG = gram_matrix(a_G)
    
    J_style_layer = tf.reduce_sum(tf.square(GS - GG))/(4.0 *(( n_H * n_W * n_C)**2))
       
    return J_style_layer


# ### **Style cost function:**
def compute_style_cost(style_image_output, generated_image_output, style_layers=STYLE_LAYERS): #style_layers=STYLE_LAYERS
    J_style = 0

    a_S = style_image_output[1:]
    a_G = generated_image_output[1:]

    for i, weight in zip(range(len(a_S)), style_layers):  
        J_style_layer = compute_layer_style_cost(a_S[i], a_G[i])
        J_style += weight[1] * J_style_layer

    return J_style


# ### **Total cost function:**
def total_cost(J_content, J_style, alpha, beta):
    t_cost = alpha * J_content + beta * J_style
    return t_cost


# In[32]:
def tensor_to_image(tensor):
    tensor = tensor * 255
    tensor = np.array(tensor, dtype=np.uint8)
    if np.ndim(tensor) > 3:
        assert tensor.shape[0] == 1
        tensor = tensor[0]

    return cv2.cvtColor(tensor, cv2.COLOR_BGR2RGB)


# In[33]:


def resize_image(image, width, height):
    if max(width,height) > 500:
      result=cv2.resize(image, (width//2, height//2), interpolation = cv2.INTER_CUBIC)
    else:
      result=image
    return result

def train_model(): 
    content_image = load_image(CONTENT_IMAGE_URL, IMG_WIDTH, IMG_HEIGHT, cv2.COLOR_BGR2RGB) 
    style_image = load_image(STYLE_IMAGE_URL, IMG_WIDTH, IMG_HEIGHT, cv2.COLOR_BGR2RGB)
      
    @tf.function()
    def train_step(generated_image, alpha, beta):
    
        with tf.GradientTape() as tape:
    
            a_G = model(generated_image)
    
            J_style = compute_style_cost(a_S, a_G)
            J_content = compute_content_cost(a_C, a_G)
    
            J = total_cost(J_content, J_style, alpha, beta)
            #J += TOTAL_VARIATION_WEIGHT * tf.image.total_variation(generated_image)   #it should reduce the noise in the generated image
            
            
        grad = tape.gradient(J, generated_image)
    
        optimizer.apply_gradients([(grad, generated_image)])
        generated_image.assign(tf.clip_by_value(generated_image, clip_value_min=0.0, clip_value_max=1.0))
    
        return J
        
    LEARNING_RATE = 0.03
    ALPHA = 1 
    BETA =  0.3 
    EPOCHS = 2000 #2000 reduced to save time in streamlit
    EPOCHS_PER_IMAGE_OUTPUT = EPOCHS//5 
    
    ADD_NOISE = True
    NOISE_RANGE = 0.2
    
    generated_image = tf.Variable(tf.image.convert_image_dtype(content_image, tf.float32))
    #model.summary()
    
    a_C = model(content_image)
    a_S = model(style_image)
    a_G = model(generated_image)
    
    optimizer = tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE)   

  
  
   
    noise_id=0
    if ADD_NOISE:
        generated_image = add_noise(generated_image, NOISE_RANGE)
        noise_id= NOISE_RANGE             

    #lst loss 
    #lst_loss=[]

    for i in range(EPOCHS):
        loss=train_step(generated_image, ALPHA, BETA)
        #loss_fmt="{:.3f}".format(loss)
        
        #lst loss 
        #lst_loss.append(np.round(loss,3))

        if i % EPOCHS_PER_IMAGE_OUTPUT == 0:
            #print(f"\n Epoch {i} \n{loss_fmt} ")
            image = tensor_to_image(generated_image)           

            cv2.imwrite(path + 'tmp_style_transfer.jpg', image)

        if  i == EPOCHS - 1:
            image = tensor_to_image(generated_image)   
            #print(f"\n Epoch {i} \n{loss_fmt} ")
            cv2.imwrite(path + 'tmp_style_transfer.jpg', image)
            #imagergb=cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            #plot_comparison(imagergb)
            

    #df_train=generate_df_loss(lst_loss, EPOCHS, LEARNING_RATE,ALPHA, BETA, ADD_NOISE,NOISE_RANGE)


    #save model
    #model.save(path_model + "vgg19_model.h5")


    return #lst_loss,df_train


# In[39]:


def style_transfer():
    
    train_model()

    return
