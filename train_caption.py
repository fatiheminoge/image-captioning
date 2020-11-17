import pickle
import keras
import numpy as np

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
            desc = 'startseq' + ' '.join(image_desc) + ' endseq'
            descriptions[image_id].append(desc)
    return descriptions

def load_photo_features(filename,dataset):
    all_features = pickle.load(open(filename,'rb'))
    features = {k: all_features[k] for k in dataset}
    return features

filename = 'Flicker8k_text/Flickr_8k.trainImages.txt'
train = load_set(filename)
print('Dataset: %d' % len(train))

train_descriptions = load_clean_descriptions('descriptions.txt',train)
print('Descriptions: train=%d' % len(train_descriptions))

train_features = load_photo_features('features.pkl',train)
print('Photos: train=%d' % len(train_features))

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

def create_sequences(tokenizer,max_len,descriptions,photos,vocab_size):
    X1,X2,y = list(),list(),list()
    
    for key,desc_list in descriptions.items():
        seq = tokenizer.text_to_sequences([desc])[0]
        for i in range(1,len(seq)):
            in_seq, out_seq = seq[:i],seq[i:]
            in_seq = keras.preprocessing.sequence.pad_sequences([in_seq],maxlen=max_len)[0]
            out_seq = keras.preprocessing.utils.to_categorical([out_seq],num_classes=vocab_size)[0]

            X1.append(photos[key][0])
            X2.append(in_seq)
            y.append(out_seq)
        return np.array(X1),np.array(X2),np.array(y)

def max_length(descriptions):
    lines = to_lines(descriptions)
    return max(len(d.split()) for d in lines)

