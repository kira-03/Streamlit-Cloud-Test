import streamlit as st
import google.generativeai as genai
import json
import pandas as pd
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
import tempfile
import os
import time
from google.cloud import storage, secretmanager
import io

class GeminiResearchChatbot:
    def __init__(self):
        # Initialize Google Cloud clients
        self.storage_client = storage.Client()
        self.secret_manager_client = secretmanager.SecretManagerServiceClient()

        # Fetch secrets from Google Cloud Secret Manager
        self.bucket_name = self.get_secret("GCS_BUCKET_NAME")
        self.api_keys = [
            self.get_secret("GEMINI_API_KEY_1"),
            self.get_secret("GEMINI_API_KEY_2"),
            self.get_secret("GEMINI_API_KEY_3"),
            self.get_secret("GEMINI_API_KEY_4")
        ]

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

        self.prompt = """You are a research assistant analyzing a scientific paper. Using only the following context, provide a clear answer to the question. Limit your response to just the specific information needed, without any additional explanation. If value is asked then return only the value with its unit. The question has two parts part 1: actual question part 2: Either String or Integer/Float once the answer is fetched, depending upon the part 2, modify the answer accordingly before returning ..."""

    def get_secret(self, secret_name):
        """Fetch a secret from Google Cloud Secret Manager"""
        project_id = os.environ.get("GCP_PROJECT_ID")
        secret_path = f"projects/{project_id}/secrets/{secret_name}/versions/latest"
        response = self.secret_manager_client.access_secret_version(request={"name": secret_path})
        return response.payload.data.decode("UTF-8")

    def set_current_api_key(self):
        genai.configure(api_key=self.api_keys[self.current_key_index])

    def list_pdf_files(self):
        """List all PDF files in the storage bucket"""
        blobs = self.storage_client.list_blobs(self.bucket_name, prefix="pdfs/")
        return [blob.name for blob in blobs if blob.name.lower().endswith('.pdf')]

    def download_pdf_from_storage(self, blob_name):
        """Download a PDF file from storage and return it as a bytes object"""
        blob = self.bucket.blob(blob_name)
        pdf_content = blob.download_as_bytes()
        return io.BytesIO(pdf_content)

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
            return ""

def main():
    st.set_page_config(page_title="Research Paper Comparison", layout="wide")
    st.markdown("""
    <style>
        body { background-color: #121212; color: white; }
    </style>
    """, unsafe_allow_html=True)

    st.title("Research Paper Comparison")
    chatbot = GeminiResearchChatbot()

    # Initialize session states
    if "questions" not in st.session_state:
        st.session_state["questions"] = []
        st.session_state["types"] = []
    if "question_added" not in st.session_state:
        st.session_state.question_added = False

    # Question input section
    col1, col2 = st.columns([3, 1])
    with col1:
        new_question = st.text_input("Enter a question:")
    with col2:
        answer_type = st.selectbox("Select answer type:", ["String", "Integer/Float"])

    if st.button("Add Question"):
        if new_question.strip():
            st.session_state["questions"].append(new_question.strip())
            st.session_state["types"].append(answer_type)
            st.success(f"✓ Question added: {new_question}")
            st.session_state.question_added = True

    if st.session_state["questions"]:
        st.write("### Current Questions:")
        for i, (q, t) in enumerate(zip(st.session_state["questions"], st.session_state["types"])):
            st.write(f"{i+1}. {q} ({t})")

    # PDF processing section
    stored_files = chatbot.list_pdf_files()
    if stored_files:
        selected_files = st.multiselect("Select PDFs from storage", stored_files)

        if selected_files and st.session_state["questions"]:
            if st.button("Process Selected PDFs"):
                all_results = []
                with st.spinner("Processing PDFs..."):
                    progress_bar = st.progress(0)

                    for i, file_name in enumerate(selected_files):
                        pdf_content = chatbot.download_pdf_from_storage(file_name)
                        if chatbot.load_pdf(pdf_content, from_storage=True):
                            results_for_pdf = {"File": file_name}

                            for question, q_type in zip(st.session_state["questions"], st.session_state["types"]):
                                column_title = question.split(":", 1)[0].strip() if ":" in question else question.strip()
                                full_question = f"{question} ({q_type})"
                                answer = chatbot.ask_question(full_question)
                                results_for_pdf[column_title] = f"{answer}" if answer else "N/A"

                            all_results.append(results_for_pdf)

                        progress_bar.progress((i + 1) / len(selected_files))

                if all_results:
                    st.write("### Results:")
                    df_results = pd.DataFrame(all_results)
                    st.dataframe(df_results)

                    csv = df_results.to_csv(index=False)
                    st.download_button(
                        label="Download Results as CSV",
                        data=csv,
                        file_name="research_results.csv",
                        mime="text/csv"
                    )
    else:
        st.info("No PDFs found in storage. Please contact your administrator to upload PDF files.")

if __name__ == "__main__":
    main()
