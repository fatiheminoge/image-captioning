import pickle
import keras
import numpy as np

import tensorflow as tf
physical_devices = tf.config.list_physical_devices('GPU') 
config = tf.compat.v1.ConfigProto()
config.gpu_options.allow_growth = True

import os
os.environ['TF_CPP_VMODULE'] = '2' 
os.environ['asm_compiler'] = '2' 

def extract_features(filename):
    model = keras.applications.VGG16()
    model = keras.Model(inputs=model.inputs, outputs=model.layers[-2].output)

    image = keras.preprocessing.image.load_img(filename,target_size=(224,224))
    image = keras.preprocessing.image.img_to_array(image)
    image = image.reshape((1,*image.shape))
    image = keras.applications.vgg16.preprocess_input(image)

    feature = model.predict(image,verbose=0)
    
    return feature

def word_for_id(integer,tokenizer):
    for word,index in tokenizer.word_index.items():
        if index == integer:
            return word
    
    return None

def generate_desc(model, tokenizer, photo, max_length):
    in_text = 'startseq'

    for i in range(max_length):
        sequence = tokenizer.texts_to_sequences([in_text])[0]
        sequence = keras.preprocessing.sequence.pad_sequences([sequence],maxlen=max_length)

        yhat = model.predict([photo,sequence],verbose=0)
        yhat = np.argmax(yhat)

        word = word_for_id(yhat,tokenizer)

        if word is None:
            break

        in_text += ' ' + word
        if word == 'endseq':
            break
    
    return in_text

def imgPrediction(filepath):
    tokenizer = pickle.load(open('tokenizer.pkl','rb'))
    max_length = 34

    model = keras.models.load_model('models\model-ep002-loss3.865-val_loss3.945.h5')
    photo = extract_features(filepath)

    description = generate_desc(model,tokenizer,photo,max_length)
    description = ' '.join([w for w in description.split()[1:-1]])
    return description