import os
import pandas as pd
import pickle
from pypdf import PdfReader
import re
import streamlit as st

# Load models
word_vector = pickle.load(open("tfidf.pkl", "rb"))
model = pickle.load(open("model.pkl", "rb"))

# Function to clean resume text
def cleanResume(txt):
    cleanText = re.sub('http\S+\s', ' ', txt)
    cleanText = re.sub('RT|cc', ' ', cleanText)
    cleanText = re.sub('#\S+\s', ' ', cleanText)
    cleanText = re.sub('@\S+', '  ', cleanText)  
    cleanText = re.sub('[%s]' % re.escape("""!"#$%&'()*+,-./:;<=>?@[\]^_{|}~"""), ' ', cleanText)
    cleanText = re.sub(r'[^\x00-\x7f]', ' ', cleanText) 
    cleanText = re.sub('\s+', ' ', cleanText)
    return cleanText

# Category mapping dictionary
category_mapping = {
    15: "Java Developer",
    23: "Testing",
    8: "DevOps Engineer",
    20: "Python Developer",
    24: "Web Designing",
    12: "HR",
    13: "Hadoop",
    3: "Blockchain",
    10: "ETL Developer",
    18: "Operations Manager",
    6: "Data Science",
    22: "Sales",
    16: "Mechanical Engineer",
    1: "Arts",
    7: "Database",
    11: "Electrical Engineering",
    14: "Health and fitness",
    19: "PMO",
    4: "Business Analyst",
    9: "DotNet Developer",
    2: "Automation Testing",
    17: "Network Security Engineer",
    21: "SAP Developer",
    5: "Civil Engineer",
    0: "Advocate",
}

# Function to categorize resumes
def categorize_resumes(uploaded_files, output_directory):
    if not os.path.exists(output_directory):
        os.makedirs(output_directory)
    
    results = []
    
    for uploaded_file in uploaded_files:
        if uploaded_file.name.endswith('.pdf'):
            reader = PdfReader(uploaded_file)
            page = reader.pages[0]
            text = page.extract_text()
            cleaned_resume = cleanResume(text)

            input_features = word_vector.transform([cleaned_resume])
            prediction_id = model.predict(input_features)[0]
            category_name = category_mapping.get(prediction_id, "Unknown")
            
            category_folder = os.path.join(output_directory, category_name)
            
            if not os.path.exists(category_folder):
                os.makedirs(category_folder)
            
            target_path = os.path.join(category_folder, uploaded_file.name)
            with open(target_path, "wb") as f:
                f.write(uploaded_file.getbuffer())
            
            results.append({'filename': uploaded_file.name, 'category': category_name})
    
    results_df = pd.DataFrame(results)
    return results_df

# Custom CSS for enhanced UI
st.markdown(f"""
    <style>
        body {{
            background-image: url('https://images.pexels.com/photos/5673502/pexels-photo-5673502.jpeg?auto=compress&cs=tinysrgb&w=1260&h=750&dpr=1');
            background-size: cover;
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
        }}
        .stApp {{
            background: rgba(0, 0, 0, 0.7);
            padding: 20px;
            border-radius: 10px;
            box-shadow: 0px 4px 15px rgba(0, 0, 0, 0.2);
            color: #ffffff;
        }}
        .stButton>button {{
            background: linear-gradient(135deg, #ff7e5f, #feb47b);
            color: white;
            border: none;
            border-radius: 50px;
            padding: 10px 20px;
            margin: 10px 0;
            font-weight: bold;
            transition: all 0.3s ease;
        }}
        .stButton>button:hover {{
            transform: scale(1.05);
            background: linear-gradient(135deg, #feb47b, #ff7e5f);
        }}
        .stTextInput>div>div>input {{
            background-color: rgba(255, 255, 255, 0.9);
            color: black;
            border: 1px solid #ddd;
            border-radius: 10px;
            padding: 10px;
        }}
        .stFileUploader>div>div>input {{
            background-color: rgba(255, 255, 255, 0.2);
            color: white;
            border: 1px solid #ddd;
            border-radius: 10px;
            padding: 10px;
        }}
        h1, h2, h3, h4, h5, h6 {{
            text-align: center;
            color: #ffffff;
            font-weight: 700;
        }}
        .css-1aumxhk {{
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            color: black;
        }}
        .stSidebar {{
            background-color: rgba(255, 255, 255, 0.1);
            padding: 20px;
            border-radius: 10px;
        }}
        .stDataFrame {{
            background-color: rgba(255, 255, 255, 0.1);
            color: white;
            border-radius: 10px;
            padding: 20px;
            box-shadow: 0px 4px 15px rgba(0, 0, 0, 0.2);
        }}
        .stSuccess {{
            background-color: black;
            color: white;
            font-size: 1.2em;
            font-weight: bold;
        }}
        .stDownloadButton {{
            font-size: 1.2em;
            font-weight: bold;
            color: black;
        }}
        label {{
            color: white !important;
            font-weight: bold !important;
        }}
    </style>
""", unsafe_allow_html=True)

# App title and subtitle
st.title("Resume Categorizer Application")
st.subheader("With Python & Machine Learning")

# File uploader and output directory input
uploaded_files = st.file_uploader("Choose PDF files", type="pdf", accept_multiple_files=True)
output_directory = st.text_input("Output Directory", r"C:\Users\SAR\Documents\output_category")

# Button to trigger the categorization
if st.button("Categorize Resumes"):
    if uploaded_files and output_directory:
        results_df = categorize_resumes(uploaded_files, output_directory)
        st.write(results_df)
        results_csv = results_df.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="Download results as CSV",
            data=results_csv,
            file_name='categorized_resumes.csv',
            mime='text/csv',
            help="Click to download the results as a CSV file.",
        )
        st.success("Resumes categorization and processing completed.", icon="✅")
    else:
        st.error("Please upload files and specify the output directory.")
