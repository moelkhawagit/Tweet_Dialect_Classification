
import pandas as pd 
import nltk
nltk.download('punkt')
nltk.download('stopwords')
from nltk.corpus import stopwords
import string
import re 
import gensim
    

def extract_Arabic_Doc_Embeddings( docSeries,  embeddingDim,  vocabMinFrequency,  trainEpochs,  file_name = 'doc_embeddings.csv', modelname = ""):
# Tokenizing the document series

    tokenized_docs = docSeries.apply(nltk.word_tokenize)
    print("First five Documents Tokenized:")
    print(tokenized_docs[:6])

# REMOVE ARABIC STOPWORDS AND NON ALPHABETIC WORDS 
    
    stop_words = set(stopwords.words('arabic'))
    for index, value in tokenized_docs.items():
        for word in value:
            if (not(word.isalpha())):
                value.remove(word) 
    print(tokenized_docs[:6])

#word in stop_words or
    
# REMOVE ENGLISH AND UNRELEVENT WORDS

    for i in range(5):
        for index, value in tokenized_docs.items():
            for w in value:
                if re.search('[a-zA-Z]', w) or re.search("[._@z'+#-?&!/,]", w):
                    value.remove(w)
    print(tokenized_docs[30:40])


#CONSTRUCT TAGGED DOCUMENT FOR MODEL TRAINING

    if modelname != "":
         model = gensim.models.doc2vec.Doc2Vec.load(modelname)
    else:
        tokenized_docs_df =  pd.DataFrame(tokenized_docs)
        taggedlist = [gensim.models.doc2vec.TaggedDocument( row[1], [row[0]]) for row in tokenized_docs_df.itertuples()]
        model = gensim.models.doc2vec.Doc2Vec(vector_size=embeddingDim, min_count=vocabMinFrequency, epochs=trainEpochs)
        model.build_vocab(taggedlist)
        print("Dictionary Size:")
        print(len(model.wv.vocab))
        model.train(taggedlist, total_examples=model.corpus_count, epochs=model.epochs)


#INFER EMBEDDINGS FOR EACH DOCUMENT 
    
    doc2vec_embedding_list = []
    for index, value in tokenized_docs.items():
        doc2vec_embedding_list.append(model.infer_vector(value))
    print("First Document Embedding")
    print(doc2vec_embedding_list[:1])
    exported_df = pd.DataFrame(doc2vec_embedding_list)
    exported_df.to_csv(file_name,index=False)
    print(f"{file_name} SAVED IN DIRECTORY")



