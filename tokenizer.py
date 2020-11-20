import keras
import pickle

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

def load_clean_description(filename,dataset):
    doc = load_doc(filename)
    descriptions = dict()
    for line in doc.split('\n'):
        tokens = line.split()
        image_id, image_desc = tokens[0],tokens[1:]

        if image_id in dataset:
            if image_id not in descriptions:
                descriptions[image_id] = list()

            desc = 'startseq ' + ' '.join(image_desc) + ' endseq'
            descriptions[image_id].append(desc)
    
    return descriptions

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

filename = 'Flicker8k_text/Flickr_8k.trainImages.txt'
train = load_set(filename)
print('Dataset: %d' % len(train))

train_descriptions = load_clean_description('descriptions.txt',train)
print('Descriptions: train=%d' % len(train_descriptions))

tokenizer = create_tokenizer(train_descriptions)
pickle.dump(tokenizer, open('tokenizer.pkl','wb'))