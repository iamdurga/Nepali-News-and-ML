import streamlit as st
from tensorflow import keras
import numpy as np    
from keras.preprocessing import sequence
import pickle

st.title("News Classification")
classes = {'news': 0,
 'business': 1,
 'world': 2,
 'sports': 3,
 'national': 4,
 'international': 5,
 'technology': 6,
 'entertainment': 7}
@st.cache
def get_stop_punctuation():
    stop_file = "nepali_stopwords.txt"
    stop_words = []
    with open(stop_file, encoding='utf-8') as fp:
        lines = fp.readlines()
        stop_words =list( map(lambda x:x.strip(), lines))
    
    punctuation_file = "nepali_punctuation (1).txt"
    punctuation_words = []
    with open(punctuation_file) as fp:
        lines = fp.readlines()
        punctuation_words =list( map(lambda x:x.strip(), lines))
    
    return stop_words, punctuation_words

@st.cache
def preprocess_text(cat_data, stop_words, punctuation_words):
  new_cat = []
  noise = "1,2,3,4,5,6,7,8,9,0,०,१,२,३,४,५,६,७,८,९".split(",")
  
  for row in cat_data:
    words = row.strip().split(" ")      
    nwords = "" # []
    
    for word in words:
      if word not in punctuation_words and word not in stop_words:
        is_noise = False
        for n in noise:
          #print(n)
          if n in word:
            is_noise = True
            break
        if is_noise == False:
          word = word.replace("(","")
          word = word.replace(")","")
          # nwords.append(word)
          if len(word)>1:
            nwords+=word+" "
          
    new_cat.append(nwords.strip())
  # print(new_cat)
  return new_cat

@st.cache(allow_output_mutation=True)
def get_model():
    model1 = keras.models.load_model('./simple_nn.h5')
    

    with open('./vectorizer.pkl', 'rb') as f:
        vect = pickle.load(f)
    return model1, vect

stop_words,punctuation_words = get_stop_punctuation()

st.markdown("## Testing Preprocess")
txt=["शिक्षण संस्थामा ज जनस्वास्थ्य 50 मापदण्ड पालना शिक्षा मन्त्रालयको निर्देशन"]
st.write(f"Preprocessing: {txt}")
title_clean = preprocess_text(txt, 
stop_words, punctuation_words)
st.write(title_clean)

st.markdown("## Loading Model")
model, vectorizer = get_model()
st.write("Done")

news = st.text_input("Enter News Title", value="")
if news:
    st.write(f"Preprocessing: {news}")
    news = preprocess_text([news], stop_words, punctuation_words)
    st.write(f"Preprocessed: {news}")    
    title_clean = vectorizer.transform(news)

    encoded_string = sequence.pad_sequences(title_clean.toarray(), maxlen=8587)

    if st.button("Predict"):
        # # Predict string class
        string_predict = model.predict(encoded_string)

        rclasses = {v:k for k,v in classes.items()}
        res = [rclasses[c] for c in np.argmax(string_predict, axis=1)]
        res = {k:v for k,v in zip(news, res)}
        st.write(res)

        



        