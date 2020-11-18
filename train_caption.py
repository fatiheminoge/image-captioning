import pickle

import keras
import numpy as np
from nltk.translate.bleu_score import corpus_bleu,SmoothingFunction

import tensorflow as tf
physical_devices = tf.config.list_physical_devices('GPU') 
tf.config.experimental.set_memory_growth(physical_devices[0], True)

def load_doc(filename):
    with open(filename,'r') as f:
        text = f.read()
    return text

def load_set(filename):
    doc = load_doc(filename)
    dataset = list()
    for line in doc.split('\n'):
        if len(line) < 1:
            continue
    
        id = line.split('.')[0]
        dataset.append(id)
    return set(dataset)

def load_clean_descriptions(filename,dataset):
    doc = load_doc(filename)
    descriptions = dict()
    for line in doc.split('\n'):
        tokens = line.split()
        image_id,image_desc = tokens[0],tokens[1:]
        if image_id in dataset:
            if image_id not in descriptions:
                descriptions[image_id] = list()
            desc = 'startseq ' + ' '.join(image_desc) + ' endseq'
            descriptions[image_id].append(desc)
    return descriptions

def load_photo_features(filename,dataset):
    all_features = pickle.load(open(filename,'rb'))
    features = {k: all_features[k] for k in dataset}
    return features

def to_lines(descriptions):
    all_desc = list()
    for key in descriptions.keys():
        [all_desc.append(d) for d in descriptions[key]]
    return all_desc

def create_tokenizer(descriptions):
    lines = to_lines(descriptions)
    tokenizer = keras.preprocessing.text.Tokenizer()
    tokenizer.fit_on_texts(lines)
    return tokenizer
    
def max_length(descriptions):
    lines = to_lines(descriptions)
    return max(len(d.split()) for d in lines)

def create_sequences(tokenizer,max_len,descriptions,photos,vocab_size):
    X1,X2,y = list(),list(),list()
    
    for key,desc_list in descriptions.items():
        for desc in desc_list:
            seq = tokenizer.texts_to_sequences([desc])[0]
            for i in range(1,len(seq)):
                in_seq, out_seq = seq[:i],seq[i]
                in_seq = keras.preprocessing.sequence.pad_sequences([in_seq],maxlen=max_len)[0]
                out_seq = keras.utils.to_categorical([out_seq],num_classes=vocab_size)[0]

                X1.append(photos[key][0])
                X2.append(in_seq)
                y.append(out_seq)
        return np.array(X1),np.array(X2),np.array(y)

def define_model(vocab_size,max_length):
    inputs1 = keras.layers.Input(shape=(4096,))
    fe1 = keras.layers.Dropout(0.5)(inputs1)
    fe2 = keras.layers.Dense(256,activation='relu')(fe1)

    inputs2 = keras.layers.Input(shape=(max_length,))
    se1 = keras.layers.Embedding(vocab_size,256,mask_zero=True)(inputs2)
    se2 = keras.layers.Dropout(0.5)(se1)
    se3 = keras.layers.LSTM(256)(se2)
    
    decoder1 = keras.layers.add([fe2,se3])
    decoder2 = keras.layers.Dense(256,activation='relu')(decoder1)
    outputs = keras.layers.Dense(vocab_size,activation='softmax')(decoder2)

    model = keras.Model(inputs=[inputs1,inputs2],outputs=outputs)
    model.compile(loss='categorical_crossentropy',optimizer='adam')

    print(model.summary())
 #   keras.utils.plot_model(model, to_file='model.png',show_shapes=True)
    return model

##############################################

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
        if word=='endseq':
            break
    return in_text

def evaluate_model(model, descriptions, photos, tokenizer, max_length):
    actual,predicted = list(),list()

    for key,desc_list in descriptions.items():
        yhat = generate_desc(model,tokenizer,photos[key],max_length)
        references = [d.split() for d in desc_list]
        actual.append(references)
        predicted.append(yhat.split())
    
    cc = SmoothingFunction()
    print('BLEU-1: %f' % corpus_bleu(actual,predicted,weights=(1.0,0,0,0),smoothing_function=cc.method3))
    print('BLEU-1: %f' % corpus_bleu(actual,predicted,weights=(0.5,0.5,0,0),smoothing_function=cc.method3))
    print('BLEU-1: %f' % corpus_bleu(actual,predicted,weights=(0.3,0.3,0.3,0),smoothing_function=cc.method3))
    print('BLEU-1: %f' % corpus_bleu(actual,predicted,weights=(0.25,0.25,0.25,0.25),smoothing_function=cc.method3))


filename = 'Flicker8k_text/Flickr_8k.trainImages.txt'
train = load_set(filename)
print('Dataset: %d' % len(train))

train_descriptions = load_clean_descriptions('descriptions.txt',train)
print('Descriptions: train=%d' % len(train_descriptions))

train_features = load_photo_features('features.pkl',train)
print('Photos: train=%d' % len(train_features))

tokenizer = create_tokenizer(train_descriptions)
vocab_size = len(tokenizer.word_index) + 1
print('Vocabulary Size: %d' % vocab_size)

max_length = max_length(train_descriptions)
print('Max Length: %d' % max_length)

X1train, X2train, ytrain = create_sequences(tokenizer, max_length, train_descriptions, train_features, vocab_size)

###################################################

filename = 'Flicker8k_text/Flickr_8k.devImages.txt'
test = load_set(filename)
print('Dataset: %d' % len(test))

test_descriptions = load_clean_descriptions('descriptions.txt',test)
print('Descriptions: test=%d' % len(test_descriptions))

test_features = load_photo_features('features.pkl',test)
print('Photos: test=%d' % len(test_features))

X1test, X2test, ytest = create_sequences(tokenizer, max_length, test_descriptions, test_features, vocab_size)

model = define_model(vocab_size,max_length)
filepath = 'models/model-ep{epoch:03d}-loss{loss:.3f}-val_loss{val_loss:.3f}.h5'
checkpoint = keras.callbacks.ModelCheckpoint(filepath,monitor='val_loss',verbose=1,save_best_only=True,mode='min')

model.fit([X1train, X2train],ytrain, epochs=20, verbose=2, callbacks=[checkpoint], validation_data=([X1test,X2test],ytest))

filename = 'models\model-ep016-loss3.095-val_loss7.944.h5'
model = keras.models.load_model(filename)

evaluate_model(model,test_descriptions,test_features,tokenizer,max_length)
