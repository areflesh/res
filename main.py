import streamlit as st 
from pdfminer.converter import TextConverter
from pdfminer.pdfinterp import PDFPageInterpreter
from pdfminer.pdfinterp import PDFResourceManager
from pdfminer.layout import LAParams
from pdfminer.pdfpage import PDFPage
import io
import os
import docx2txt
import spacy
from spacy.matcher import Matcher
import pandas as pd
import numpy as np
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords 
import nltk
import plotly.graph_objects as go
nltk.download('punkt')
nltk.download('stopwords')
# load pre-trained model
nlp = spacy.load('en_core_web_sm')

# initialize matcher with a vocab
matcher = Matcher(nlp.vocab)

        
def extract_skills(resume_text):
    nlp_text = nlp(resume_text)

    # removing stop words and implementing word tokenization
    tokens = [token.text for token in nlp_text if not token.is_stop]
    
    # reading the csv file
    data = pd.read_csv("skills.csv") 
    
    # extract values
    #skills= list(data["Skill"].values)
    #print(skills)
    skillset = {"Tech":[],"Python":[],"ML":[],"DL":[],"NLP":[],"Non tech":[]}

    # check for one-grams (example: python)
    for token in tokens:
        if data[data["Skill"]==token.lower()]["Group"].str.contains("0").any():
            skillset["Tech"].append(token)
        elif data[data["Skill"]==token.lower()]["Group"].str.contains("1").any():
            skillset["Python"].append(token)
        elif data[data["Skill"]==token.lower()]["Group"].str.contains("2").any():
            skillset["ML"].append(token)
        elif data[data["Skill"]==token.lower()]["Group"].str.contains("3").any():
            skillset["DL"].append(token)
        elif data[data["Skill"]==token.lower()]["Group"].str.contains("4").any():
            skillset["NLP"].append(token)
        elif data[data["Skill"]==token.lower()]["Group"].str.contains("9").any():
            skillset["Non tech"].append(token)
    
    # check for bi-grams and tri-grams (example: machine learning)
    for token in nlp_text.noun_chunks:
        token = token.text.lower().strip()
        if data[data["Skill"]==token]["Group"].str.contains("0").any():
            skillset["Tech"].append(token)
        elif data[data["Skill"]==token]["Group"].str.contains("1").any():
            skillset["Python"].append(token)
        elif data[data["Skill"]==token]["Group"].str.contains("2").any():
            skillset["ML"].append(token)
        elif data[data["Skill"]==token]["Group"].str.contains("3").any():
            skillset["DL"].append(token)
        elif data[data["Skill"]==token]["Group"].str.contains("4").any():
            skillset["NLP"].append(token)
        elif data[data["Skill"]==token]["Group"].str.contains("9").any():
            skillset["Non tech"].append(token)
    
    # double check tokenazation using nltk 
    tokenized = word_tokenize(resume_text)
    stop_words = set(stopwords.words('english')) 
    tokens = [w for w in tokenized if not w in stop_words]
    
    for token in tokens:
        if data[data["Skill"]==token.lower()]["Group"].str.contains("0").any():
            skillset["Tech"].append(token)
        elif data[data["Skill"]==token.lower()]["Group"].str.contains("1").any():
            skillset["Python"].append(token)
        elif data[data["Skill"]==token.lower()]["Group"].str.contains("2").any():
            skillset["ML"].append(token)
        elif data[data["Skill"]==token.lower()]["Group"].str.contains("3").any():
            skillset["DL"].append(token)
        elif data[data["Skill"]==token.lower()]["Group"].str.contains("4").any():
            skillset["NLP"].append(token)
        elif data[data["Skill"]==token.lower()]["Group"].str.contains("9").any():
            skillset["Non tech"].append(token)
    
    # check for bi-grams and tri-grams (example: machine learning)
    for token_or in nltk.bigrams(tokens):
        token = " ".join(token_or).lower().strip()
        if data[data["Skill"]==token]["Group"].str.contains("0").any():
            skillset["Tech"].append(token)
        elif data[data["Skill"]==token]["Group"].str.contains("1").any():
            skillset["Python"].append(token)
        elif data[data["Skill"]==token]["Group"].str.contains("2").any():
            skillset["ML"].append(token)
        elif data[data["Skill"]==token]["Group"].str.contains("3").any():
            skillset["DL"].append(token)
        elif data[data["Skill"]==token]["Group"].str.contains("4").any():
            skillset["NLP"].append(token)
        elif data[data["Skill"]==token]["Group"].str.contains("9").any():
            skillset["Non tech"].append(token) 
    for key in skillset.keys():
        skillset[key]=[i.capitalize() for i in set([i.lower() for i in skillset[key]])]
    return skillset

def extract_name(resume_text):
    nlp_text = nlp(resume_text)
    # First name and Last name are always Proper Nouns
    pattern = [{'POS': 'PROPN'}, {'POS': 'PROPN'}]
    matcher.add('NAME', [pattern])
    matches = matcher(nlp_text)
    for match_id, start, end in matches:
        span = nlp_text[start:end]
        return span.text

def extract_text_from_doc(doc_path):
    temp = docx2txt.process(doc_path)
    text = [line.replace('\t', ' ') for line in temp.split('\n') if line]
    return ' '.join(text)

def extract_text_from_pdf(pdf_path):
    with open(pdf_path, 'rb') as fh:
        # iterate over all pages of PDF document
        for page in PDFPage.get_pages(fh, caching=True, check_extractable=True):
            # creating a resoure manager
            resource_manager = PDFResourceManager()
            
            # create a file handle
            fake_file_handle = io.StringIO()
            
            # creating a text converter object
            converter = TextConverter(
                                resource_manager, 
                                fake_file_handle, 
                                #codec='utf-8', 
                                laparams=LAParams()
                        )

            # creating a page interpreter
            page_interpreter = PDFPageInterpreter(
                                resource_manager, 
                                converter
                            )

            # process current page
            page_interpreter.process_page(page)
            
            # extract text
            text = fake_file_handle.getvalue()
            yield text

            # close open handles
            converter.close()
            fake_file_handle.close()

st.set_page_config(layout="wide")
file_list = os.listdir("./resumes")
file_list.insert(0,"None")
file_selection = st.sidebar.selectbox("Choose file",file_list)
final_data = []

if (file_selection is not None)&(file_selection!="None"):
    
        if os.path.splitext(file_selection)[1]==".pdf":
            text = ""
            for page in extract_text_from_pdf("./resumes/"+file_selection):
                text += ' ' + page
            name = extract_name(text)
            
            
            skills = extract_skills(text)
            if name!=os.path.splitext(file_selection)[0]:
                name = os.path.splitext(file_selection)[0]
            st.write(name)
            col1,col2,col3,col4,col5,col6 = st.beta_columns(6)
            col1.write("Tech skils")
            col2.write("Python")
            col3.write("ML")
            col4.write("DL")
            col5.write("NLP")
            col6.write("Non tech")
            if "Tech" in skills:
                col1.write(skills["Tech"])
            else:
                col1.write("None")
            if "Python" in skills:
                col2.write(skills["Python"])
                py_len = len(skills["Python"])
            else:
                col2.write("None")
                py_len = 0 
            if "ML" in skills:
                col3.write(skills["ML"])
                ml_len = len(skills["ML"])
            else:
                col3.write("None")
                ml_len = 0 
            if "DL" in skills:
                col4.write(skills["DL"])
                dl_len = len(skills["DL"])
            else:
                col4.write("None")
                dl_len = 0 
            if "NLP" in skills:
                col5.write(skills["NLP"])
                nlp_len = len(skills["NLP"])
            else:
                col5.write("None")
                nlp_len = 0 
            if "Non tech" in skills:
                col6.write(skills["Non tech"])
            else:
                col6.write("None") 
            fig = go.Figure([go.Bar(x=["Python", "ML", "DL","NLP"], y=[py_len*100/20,ml_len*100/20,dl_len*100/20,nlp_len*100/20])])
            fig.update_layout(title="Points by skills", width=1200)
            st.write(fig)
        else: 
            text = extract_text_from_doc("./resumes/"+file_selection)
            name = extract_name(text)
            
            
            skills = extract_skills(text)
            if name!=os.path.splitext(file_selection)[0]:
                name = os.path.splitext(file_selection)[0]
            st.write(name)
            col1,col2,col3,col4,col5,col6 = st.beta_columns(6)
            col1.write("Tech skils")
            col2.write("Python")
            col3.write("ML")
            col4.write("DL")
            col5.write("NLP")
            col6.write("Non tech")
            if "Tech" in skills:
                col1.write(skills["Tech"])
            else:
                col1.write("None")
            if "Python" in skills:
                col2.write(skills["Python"])
                py_len = len(skills["Python"])
            else:
                col2.write("None")
                py_len = 0 
            if "ML" in skills:
                col3.write(skills["ML"])
                ml_len = len(skills["ML"])
            else:
                col3.write("None")
                ml_len = 0 
            if "DL" in skills:
                col4.write(skills["DL"])
                dl_len = len(skills["DL"])
            else:
                col4.write("None")
                dl_len = 0 
            if "NLP" in skills:
                col5.write(skills["NLP"])
                nlp_len = len(skills["NLP"])
            else:
                col5.write("None")
                nlp_len = 0 
            if "Non tech" in skills:
                col6.write(skills["Non tech"])
            else:
                col6.write("None") 
            fig = go.Figure([go.Bar(x=["Python", "ML", "DL","NLP"], y=[py_len*100/20,ml_len*100/20,dl_len*100/20,nlp_len*100/20])])
            fig.update_layout(title="Points by skills", width=1200)
            st.write(fig)

