import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import streamlit as st

def set_custom_style():
    st.markdown(
        """
        <style>
        /* background color here--green */
        .stApp {
            background-color: #96c8a2 ; 
        }

        /* heading color*/
        h1{
            color:#062a78 ;
            text-align:center;
        }

        /* heading h3 blue */
        h3{
            color:#1d2951;
        }

         /* input box */
        .stTextInput > div > div > input {
            background-color: #f0f8ff;  
            color: #1d2951; /* Text color */
            border-color:yellow;
        }

        /* input box */
        .st-ay.stTextInput input:focus {
             border-color: #367588 !important;
        }


         /* uploading qna file name */
        .stFileUploaderFileName{
            color: #1d2951;
        }

        /* Icon color */
        .eyeqlp51 {
            fill: #5d89ba;
        }


        /* paragraphs */
        div[data-testid="stMarkdownContainer"] p {
            color: #002e63;
            font-weight: bold;
        }

        /* buttons -- browse files*/
        button.st-emotion-cache-7ym5gk.ef3psqc12 {
            background-color: #5f9ea0;
            color: #002e63;
            font-weight: bold;
            border: none;
            padding: 10px 20px;
            border-radius: 5px;
            cursor: pointer;
        }

        button.st-emotion-cache-7ym5gk.ef3psqc12:hover {
            background-color: #2f847c;
            color: #002e63;
        }

        /* drag and drop box*/
        .st-emotion-cache-1gulkj5 {
            background-color: #f0f8ff; 
            color:#002e63;
        }
        
        </style>
        """,
        unsafe_allow_html=True,
    )

def main():
    set_custom_style()
    st.title('QUESTION ANSWERING SYSTEM')

    uploaded_file = st.file_uploader("Choose a CSV file", type="csv")

    if uploaded_file is not None:
        try:
            df = pd.read_csv(uploaded_file)  # Read the uploaded file into DataFrame
            return df
        except Exception as e:
            st.write("Error reading CSV file:", e)
            return None
    else:
        st.write("Please upload a CSV file.")

def get_most_similar_question(new_sentence, tfidf_matrix, vectorizer):
    new_tfidf = vectorizer.transform([new_sentence])  # Use the same vectorizer instance

    similarities = cosine_similarity(new_tfidf, tfidf_matrix)

    most_similar_index = np.argmax(similarities)

    similarity_percentage = similarities[0, most_similar_index] * 100

    return most_similar_index, similarity_percentage

def AnswerTheQuestion(new_sentence, df, tfidf_matrix, vectorizer):
    if df is not None:
        questions = df['question'].tolist()

        most_similar_index, similarity_percentage = get_most_similar_question(new_sentence, tfidf_matrix, vectorizer)
        if similarity_percentage > 70:
            most_similar_answer = df.iloc[most_similar_index]['answer']
            response = {
                "answer": most_similar_answer
            }
        else:
            response = {
                "answer": "Sorry, I don't have this information"
            }
        return response
    else:
        return {"answer": "No CSV file uploaded yet"}

df = main()
if df is not None:
    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform(df['question'].tolist())

    # Streamlit UI for user input
    st.subheader("Enter your Question: ")
    user_question = st.text_input("")

    if st.button("Get Answer"):
        if user_question.strip() == "":
            st.error("Please enter a valid question.")
        else:
            response = AnswerTheQuestion(user_question, df, tfidf_matrix, vectorizer)
            st.write("Answer:", response["answer"])


