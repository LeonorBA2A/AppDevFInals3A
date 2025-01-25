import os
from docx import Document
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer
import nltk
import streamlit as st
import re

# Download NLTK data
nltk.download('punkt')
nltk.download('stopwords')

# Step 1: Read .docx files
def read_docx(file_path):
    doc = Document(file_path)
    content = []
    for paragraph in doc.paragraphs:
        content.append(paragraph.text)
    return "\n".join(content)

def load_documents(folder_path):
    documents = []
    filenames = []
    file_paths = []
    for file_name in os.listdir(folder_path):
        if file_name.endswith(".docx"):
            file_path = os.path.join(folder_path, file_name)
            documents.append(read_docx(file_path))
            filenames.append(file_name)
            file_paths.append(file_path)
    return documents, filenames, file_paths

# Step 2: Preprocess text
stop_words = set(stopwords.words("english"))
stemmer = PorterStemmer()

def preprocess_text(text):
    tokens = word_tokenize(text.lower())  # Tokenize and lowercase
    tokens = [word for word in tokens if word.isalnum()]  # Remove non-alphanumeric tokens
    tokens = [word for word in tokens if word not in stop_words]  # Remove stop words
    tokens = [stemmer.stem(word) for word in tokens]  # Apply stemming
    return " ".join(tokens)

# Step 3: Create TF-IDF vector space
def build_tfidf_model(documents):
    preprocessed_docs = [preprocess_text(doc) for doc in documents]
    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform(preprocessed_docs)
    return vectorizer, tfidf_matrix

# Step 4: Extract years of experience from text
def extract_experience(text):
    """Extract years of experience from a text using regex."""
    experience_pattern = re.compile(r"(\d+)\s*(?:\+?\s*years?|yrs?)", re.IGNORECASE)
    matches = experience_pattern.findall(text)
    years_of_experience = [int(match) for match in matches if match.isdigit()]
    return max(years_of_experience, default=0)  # Return the highest experience found or 0 if none

# Step 5: Validate against combined criteria
def validate_combined_criteria(criteria, min_experience, max_experience, vectorizer, tfidf_matrix, documents, filenames, file_paths):
    criteria_vector = vectorizer.transform([preprocess_text(criteria)])
    similarities = cosine_similarity(criteria_vector, tfidf_matrix).flatten()
    
    results = []
    for idx, similarity in enumerate(similarities):
        experience = extract_experience(documents[idx])
        if min_experience <= experience <= max_experience:
            results.append((filenames[idx], file_paths[idx], similarity * 10, experience))
    
    # Sort results by match score (desc) and years of experience (desc)
    sorted_results = sorted(results, key=lambda x: (-x[2], -x[3]))
    return sorted_results

# Streamlit App
def main():
    st.title("Automated Applicant Recommendation with Combined Criteria")
    st.sidebar.header("Settings")

    # Upload folder path
    folder_path = st.sidebar.text_input("Folder Path", r"path_to_your_folder")  # Replace with your folder

    # Validate folder path
    if folder_path and not os.path.exists(folder_path):
        st.error("Folder path does not exist. Please provide a valid path.")
        return

    # Load documents
    if st.sidebar.button("Load Documents"):
        if not folder_path:
            st.error("Please enter a folder path.")
        else:
            documents, filenames, file_paths = load_documents(folder_path)
            if not documents:
                st.error("No .docx files found in the folder.")
            else:
                st.session_state["documents"] = documents
                st.session_state["filenames"] = filenames
                st.session_state["file_paths"] = file_paths
                st.session_state["vectorizer"], st.session_state["tfidf_matrix"] = build_tfidf_model(documents)
                st.success(f"Loaded {len(documents)} documents.")

    # Combined validation
    if "documents" in st.session_state and "vectorizer" in st.session_state:
        criteria = st.sidebar.text_area("Enter Hiring Criteria (e.g., skills, qualifications):")
        min_experience = st.sidebar.number_input("Minimum Years of Experience", min_value=0, value=2, step=1)
        max_experience = st.sidebar.number_input("Maximum Years of Experience", min_value=0, value=5, step=1)
        top_n_results = st.sidebar.slider("Limit Results To Top-N", min_value=1, max_value=50, value=10)

        if st.sidebar.button("Validate Resumes"):
            results = validate_combined_criteria(
                criteria,
                min_experience,
                max_experience,
                st.session_state["vectorizer"],
                st.session_state["tfidf_matrix"],
                st.session_state["documents"],
                st.session_state["filenames"],
                st.session_state["file_paths"]
            )
            st.session_state["combined_results"] = results[:top_n_results]  # Slice to limit top-N results
            st.success("Validation completed.")

        # Display results
        if "combined_results" in st.session_state:
            st.subheader("Top-N Combined Validation Results")
            for file, path, score, experience in st.session_state["combined_results"]:
                st.write(f"**{file}** - Match Score: {score:.4f}, Experience: {experience} years")
                if st.button(f"View {file}"):
                    st.session_state["selected_file"] = path

    # Display content of the selected document
    if "selected_file" in st.session_state:
        selected_file_path = st.session_state["selected_file"]
        content = read_docx(selected_file_path)
        st.subheader(f"Content of {os.path.basename(selected_file_path)}:")
        st.text_area("", content, height=300)

if __name__ == "__main__":
    main()
