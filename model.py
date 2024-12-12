import torch
from sentence_transformers import SentenceTransformer, util
from pdfminer.high_level import extract_text


model = SentenceTransformer('sentence-transformers/gtr-t5-large')


def extract_text_from_pdf(pdf_file_path):
    try: 
        text = extract_text(pdf_file_path)
        return text
    except Exception as e:
        print(f"Error reading the PDF file: {e}")
        return ""


def calculate_similarity(resume_text, job_description):
    resume_embedding = model.encode(resume_text, convert_to_tensor=True)
    job_desc_embedding = model.encode(job_description, convert_to_tensor=True)

    cosine_sim = util.pytorch_cos_sim(resume_embedding, job_desc_embedding)
    
    return cosine_sim.item() * 100


def is_candidate_qualified(similarity_score, threshold=70):
    return similarity_score >= threshold



def main(resume_file_path, job_description):
    resume_text = extract_text_from_pdf(resume_file_path)
    if not resume_text:
        print("No resume text found. Exiting.")
        return
    
    similarity_score = calculate_similarity(resume_text, job_description)
    print(f"Resume to Job Description Similarity: {similarity_score}%")

    qualified = is_candidate_qualified(similarity_score)
    if qualified:
        print(f"Candidate is qualified for the job!")
    else:
        print(f"Candidate is not qualified for the job.")


resume_file_path = 'D:\intel\Tejasram_ats_resume_new.pdf'  
job_description = '''
Selected intern's day-to-day responsibilities include:1. NLP: Work on vector space modeling in NLP, LSTMS, sequence modeling, attention modeling, BERT, transformers2. Use the above-mentioned techniques to perform document classification, semantic similarity, NER, sentiment analysis3. Perform time series analysis using ARIMA, SARIMA, LSTMs, etc4. Work on feature engineering, feature selection/feature importance, dimensionality

Skill(s) required

Data ScienceMachine LearningNatural Language Processing (NLP)Python

Earn certifications in these skills

Learn Python
Learn Voice App Development
Learn Machine Learning
Learn Data Science

Who can apply

Only those candidates can apply who:

1. are available for the work from home job/internship

2. can start the work from home job/internship between 18th Dec'23 and 22nd Jan'24

3. are available for duration of 3 months

4. have relevant skills and interests

- Women wanting to start/restart their career can also apply.

Added requirements

1. Hands-on with Python/R programming and knowledge of machine learning tools like Scikit-Learn, Pandas, TensorFlow

2. Knowledge of statistical modeling and popular machine learning models
'''



main(resume_file_path, job_description)



