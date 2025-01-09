import streamlit as st
import google.generativeai as genai
import json
import pandas as pd
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
import tempfile
import os
import time
from google.cloud import storage
import io

class GeminiResearchChatbot:
    def __init__(self):
        # Load secrets from Streamlit's secrets manager
        self.bucket_name = st.secrets["GCS_BUCKET_NAME"]
        self.api_keys = [
            st.secrets["GEMINI_API_KEY_1"],
            st.secrets["GEMINI_API_KEY_2"],
            st.secrets["GEMINI_API_KEY_3"],
            st.secrets["GEMINI_API_KEY_4"]
        ]
        
        # Initialize Google Cloud Storage client with default credentials
        self.storage_client = storage.Client(project=st.secrets["GCP_PROJECT_ID"])
        
        self.current_key_index = 0
        self.set_current_api_key()

        self.bucket = self.storage_client.bucket(self.bucket_name)
        self.model = genai.GenerativeModel('gemini-1.5-flash')
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            length_function=len
        )
        self.documents = None

        try:
            self.semantics = self.load_semantics("Semantics.json")
        except:
            self.semantics = {}

        self.prompt = """You are a research assistant analyzing a scientific paper. Using only the following context, provide a clear answer to the question. Limit your response to just the specific information needed, without any additional explanation. If value is asked then return only the value with its unit. The question has two parts part 1: actual question part 2: Either String or Integer/Float once the answer is fetched, depending upon the part 2, modify the answer accordingly before returning

For example:
for the question metal : What are all the metals ions in the ZIF-8 compound? (String), the answer should be ZnÂ²âº,CuÂ²âº

for the question porosity_nature : What is the porous nature of the ZIF-8 compound(just specify if microporous or mesoporous or macorporous)? (String)
the answer should be either microporous or mesoporous or macorporous or Null string

for the question surface_area : What is the surface area of the ZIF-8 compound? (Integer/Float)
the answer should be like 1171.3 mÂ²/g or 1867 mÂ² gâ»Â¹ (if the answer is any other unit, then change it to the most IUPAC unit)

for the question dimension : What is the dimension of the ZIF-8 compound (say either 2D or 3D)? (String)
the answer should be either 2D or 3D or Null string

for the question morphology : What is the morphology of the ZIF-8 compound? (String)
the answer should be like leaf-shaped or rhombic dodecahedron etc.

for the question size : What is the size of the ZIF-8 compound? (Integer/Float)
the answer should be a value like 270 nm, if any other units are fetched, then change it to IUPAC unit

If the answer is not there in the pdf, then return a Null string
If a value with new unit is fetched, then try to convert it to the unit which is widely used for that question

Technical Context from Research Paper: {context}
Semantic Context: {semantic_context}
Question: {question}
Answer:"""

    def set_current_api_key(self):
        genai.configure(api_key=self.api_keys[self.current_key_index])

    def list_pdf_files(self):
        """List all PDF files in the storage bucket"""
        try:
            blobs = self.storage_client.list_blobs(self.bucket_name, prefix="pdfs/")
            return [blob.name for blob in blobs if blob.name.lower().endswith('.pdf')]
        except Exception as e:
            st.error(f"Error accessing storage bucket: {str(e)}")
            return []

    def download_pdf_from_storage(self, blob_name):
        """Download a PDF file from storage and return it as a bytes object"""
        try:
            blob = self.bucket.blob(blob_name)
            pdf_content = blob.download_as_bytes()
            return io.BytesIO(pdf_content)
        except Exception as e:
            st.error(f"Error downloading PDF: {str(e)}")
            return None

    def load_semantics(self, path):
        try:
            with open(path, "r") as file:
                semantics = json.load(file)
                processed_semantics = {}
                for category, details in semantics.items():
                    processed_semantics[category] = {
                        'attributes': details.get('attributes', []),
                        'attributes_text': ', '.join(details.get('attributes', [])),
                        'relations': details.get('relations', {}),
                        'relations_text': ', '.join([f"{rel}: {', '.join(values)}" for rel, values in details.get('relations', {}).items()])
                    }
                return processed_semantics
        except Exception as e:
            st.warning(f"Could not load semantics file: {str(e)}")
            return {}

    def expand_semantic_context(self, question):
        expanded_context = ""
        matched_categories = set()
        for category, details in self.semantics.items():
            matches = (
                category.lower() in question.lower() or
                any(attr.lower() in question.lower() for attr in details['attributes']) or
                any(keyword.lower() in question.lower() for keyword in (
                    category.split() + details['attributes'] + list(details['relations'].keys()) +
                    [item for sublist in details['relations'].values() for item in sublist]
                ))
            )
            if matches:
                matched_categories.add(category)

        if not matched_categories:
            matched_categories = set(self.semantics.keys())

        for category in matched_categories:
            details = self.semantics[category]
            expanded_context += f"\n--- {category} Semantic Context ---\n"
            expanded_context += f"Attributes: {details['attributes_text']}\n"
            expanded_context += f"Relations: {details['relations_text']}\n"

        return expanded_context

    def load_pdf(self, pdf_file, from_storage=False):
        try:
            with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as temp_file:
                if from_storage:
                    temp_file.write(pdf_file.read())
                else:
                    pdf_file.seek(0)
                    temp_file.write(pdf_file.read())
                temp_file_path = temp_file.name

            loader = PyPDFLoader(temp_file_path)
            self.documents = loader.load()

            if not self.documents:
                return False

            self.texts = self.text_splitter.split_documents(self.documents)

            return bool(self.texts)

        except Exception as e:
            st.error(f"Error loading PDF: {str(e)}")
            return False

        finally:
            if 'temp_file_path' in locals() and os.path.exists(temp_file_path):
                os.remove(temp_file_path)

    def ask_question(self, question):
        if not self.documents:
            return ""

        try:
            context = ""
            for text in self.texts:
                context += text.page_content + "\n"

            semantic_context = self.expand_semantic_context(question)

            prompt = self.prompt.format(
                context=context,
                semantic_context=semantic_context,
                question=question
            )

            while True:
                try:
                    self.set_current_api_key()
                    response = self.model.generate_content(prompt)

                    if response and response.text.strip():
                        answer = response.text.strip()
                        if "does not contain" in answer.lower() or "does not give" in answer.lower():
                            return ""
                        return answer
                    else:
                        return ""

                except Exception as e:
                    error_message = str(e)
                    if "429" in error_message:
                        self.current_key_index = (self.current_key_index + 1) % len(self.api_keys)
                        time.sleep(1)
                    else:
                        return ""

        except Exception as e:
            st.error(f"Error processing question: {str(e)}")
            return ""

def main():
    st.set_page_config(page_title="Research Paper Comparison", layout="wide")
    st.markdown("""
    <style>
        body { background-color: #121212; color: white; }
    </style>
    """, unsafe_allow_html=True)

    st.title("Research Paper Comparison")
    
    # Initialize session states
    if 'chatbot' not in st.session_state:
        try:
            st.session_state.chatbot = GeminiResearchChatbot()
        except Exception as e:
            st.error(f"Error initializing chatbot: {str(e)}")
            return

    if "questions" not in st.session_state:
        st.session_state.questions = []
        st.session_state.types = []
    if "results" not in st.session_state:
        st.session_state.results = None

    # Question input section
    col1, col2 = st.columns([3, 1])
    with col1:
        new_question = st.text_input("Enter a question:")
    with col2:
        answer_type = st.selectbox("Select answer type:", ["String", "Integer/Float"])

    if st.button("Add Question"):
        if new_question.strip():
            st.session_state.questions.append(new_question.strip())
            st.session_state.types.append(answer_type)
            st.success(f"Question added: {new_question}")

    # Display current questions
    if st.session_state.questions:
        st.write("### Current Questions:")
        for i, (q, t) in enumerate(zip(st.session_state.questions, st.session_state.types)):
            st.write(f"{i+1}. {q} ({t})")

        # Process PDFs button
        if st.button("Process PDFs"):
            stored_files = st.session_state.chatbot.list_pdf_files()
            if stored_files:
                all_results = []
                processing_container = st.empty()
                progress_bar = st.progress(0)
                total_files = len(stored_files)

                for i, file_name in enumerate(stored_files):
                    processing_container.write(f"Processing: {file_name}")
                    pdf_content = st.session_state.chatbot.download_pdf_from_storage(file_name)
                    
                    if pdf_content and st.session_state.chatbot.load_pdf(pdf_content, from_storage=True):
                        results_for_pdf = {"File": file_name}

                        for question, q_type in zip(st.session_state.questions, st.session_state.types):
                            column_title = question.split(":", 1)[0].strip() if ":" in question else question.strip()
                            full_question = f"{question} ({q_type})"
                            answer = st.session_state.chatbot.ask_question(full_question)
                            results_for_pdf[column_title] = f"{answer}" if answer else "N/A"

                        all_results.append(results_for_pdf)
                    
                    progress_bar.progress((i + 1) / total_files)

                if all_results:
                    st.session_state.results = pd.DataFrame(all_results)
                    processing_container.empty()
                    progress_bar.empty()

            else:
                st.error("No PDFs found in storage. Please ensure PDFs are uploaded to the storage bucket.")

    # Display results if they exist
    if st.session_state.results is not None:
        st.write("### Results:")
        st.dataframe(st.session_state.results)

        csv = st.session_state.results.to_csv(index=False)
        st.download_button(
            label="Download Results as CSV",
            data=csv,
            file_name="research_results.csv",
            mime="text/csv"
        )

if __name__ == "__main__":
    main()
