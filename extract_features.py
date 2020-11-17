import os
import pickle
import keras
import pickle

def extract_features(dir):
    model = keras.applications.VGG16()
    model = keras.models.Model(inputs=model.inputs, outputs=model.layers[-2].output)
    
    features = dict()
    for name in os.listdir(dir):
        file_path = os.path.join(dir,name)

        image = keras.preprocessing.image.load_img(file_path,target_size=(224,224))
        image = keras.preprocessing.image.img_to_array(image)
        image = image.reshape((1,*image.shape))
        image = keras.applications.vgg16.preprocess_input(image)

        feature = model.predict(image)
        image_id = name.split('.')[0]

        features[image_id] = feature
        print('>%s' % name)
    return features

directory = 'Flicker8k_Dataset'
features = extract_features(directory)

pickle.dump(features, open('features.pkl','wb'))