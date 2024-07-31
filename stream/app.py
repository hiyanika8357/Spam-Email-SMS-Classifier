import streamlit as st  #creating the web app
import pickle  #loading the saved model and vectorizer
from nltk.corpus import stopwords  #list of common stopwords
import nltk  #natural language processing tasks
from nltk.stem.porter import PorterStemmer  #stemming words
ps = PorterStemmer()  #creates an instance of the PorterStemmer, which will be used to stem words

def transform_text(text):
    text = text.lower()  #converts the text to lowercase
    text = nltk.word_tokenize(text)  #tokenizes the text into words
    text = [ps.stem(i) for i in text if i.isalnum() and i not in stopwords.words('english')]  #removes non-alphanumeric characters and stopwords, then stems the remaining words
    return " ".join(text)  #joins the processed words back into a single string

tfidf = pickle.load(open('vectorizer.pkl','rb'))
model = pickle.load(open('model.pkl','rb'))
#loads the pre-trained TF-IDF vectorizer and the classification model from saved files

# Custom CSS to change the background color and button size
css = '''
<style>
.stApp {
    background-color: #B0C4DE;  
}
button_style = """
    <style>
        /* Increase button size and change color */
        .stButton>button {
            width: 300px;  /* Button width */
            height: 70px;  /* Button height */
            font-size: 60px;  /* Font size */
        }
        /* Change button size and color on hover */
        .stButton>button:hover {
            transform: scale(1.2);  /* Increase size by 30% */
        }
</style>
'''
st.markdown(css, unsafe_allow_html=True)
st.title("SMS Spam Classifier")  #sets the title of the Streamlit web application
input_sms = st.text_area("Enter the message")  #creates a text area in the app for the user to input an SMS message

if st.button('Predict'):
    transformed_sms = transform_text(input_sms)
    vector_input = tfidf.transform([transformed_sms])
    result = model.predict(vector_input)[0]
    st.write(f'<h1 style="color:{"red" if result == 1 else "green"};">{"Spam" if result == 1 else "Not Spam"}</h1>', unsafe_allow_html=True)
 #When the "Predict" button is clicked ::
# The input SMS message is transformed using the transform_text function.
# The transformed text is converted to a TF-IDF vector.
# The vector is passed to the pre-trained model to get a prediction.
# The result is displayed as "Spam" or "Not Spam" with corresponding colors (red for spam, green for not spam).