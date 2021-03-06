phrasecut_path = path + '/PhraseCutDataset/'
sys.path.append(phrasecut_path)

from PIL import Image, ImageDraw
from utils.refvg_loader import RefVGLoader

from file_paths import dataset_dir, img_fpath 


############ create ground truth masks ############

def polygons_to_mask(polygons, w, h):
    p_mask = np.zeros((h, w))
    for polygon in polygons:
        if len(polygon) < 2:
            continue
        p = []
        for x, y in polygon:
            p.append((int(x), int(y)))
        img = Image.new('L', (w, h), 0)
        ImageDraw.Draw(img).polygon(p, outline=1, fill=1)
        mask = np.array(img)
        p_mask += mask
    p_mask = p_mask > 0
    return p_mask

############ VOCAB ############

# create phrasecut vocabulary for LSTM
class Vocabulary(object):
    def __init__(self): #Vocabulary initialization as empty
        self.word_to_index = {}
        self.index_to_word = {}
        self.current_index = 0
    
    #Add a word to the vocabulary 
    def add_word(self, word):
        if word not in self.word_to_index: #If word is new
            self.word_to_index[word] = self.current_index #add it to word list
            self.index_to_word[self.current_index] = word #and connect word's index to word
            self.current_index += 1 #increment index by 1
    
    def __len__(self):
        return len(self.word_to_index) #return vocabulary size

class Corpus(object):
    def __init__(self):
        self.vocabulary = Vocabulary()

    def split_and_add_words_to_vocab_from_data(self, data):
        #self.vocabulary.add_word('<unk>') #First, add a token for an unknown word
        for datatype in data:
            for pair in data[datatype]:
                line = pair[0]
                words = line.split() #split the words in it 
                for word in words: #for each of the words in that given line
                    word = word.lower() 
                    self.vocabulary.add_word(word) #add it to the vocabulary 
        self.vocabulary.add_word('<unk>') #First, add a token for an unknown word

    def tokenize_sentence(self, sentence):
        word_list = sentence.split()
        number_of_words_in_sentence = len(word_list)

        if number_of_words_in_sentence>20:
           word_list = word_list[:20]
           number_of_words_in_sentence = 20

        tokenized_sentence = torch.zeros(1, 1, 20).to(device)
        i = 20 - number_of_words_in_sentence
        
        for word in word_list: #for each of the words in that given sentence
            word = word.lower() 
            if word in self.vocabulary.word_to_index: #if that word exists in the vocabulary          
                tokenized_sentence[:, :, i] = self.vocabulary.word_to_index[word] #update the index of the word in that file position
            else: #if that word doesn't exist in the vocabulary
                tokenized_sentence[:, :, i] = self.vocabulary.word_to_index['<unk>'] #update the index of the unknown word token
            i += 1 #increment sentence position
        #print(tokenized_sentence)
        
        return tokenized_sentence
    
       
