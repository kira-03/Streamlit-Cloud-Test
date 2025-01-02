import streamlit as st
import google.generativeai as genai
import json
import pandas as pd
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
import tempfile
import os
import time

class GeminiResearchChatbot:
    def __init__(self):
        # Retrieve API keys from Streamlit secrets
        self.api_keys = [
            st.secrets["GEMINI_API_KEY_1"],
            st.secrets["GEMINI_API_KEY_2"],
            st.secrets["GEMINI_API_KEY_3"],
            st.secrets["GEMINI_API_KEY_4"]
        ]
        self.current_key_index = 0
        self.set_current_api_key()
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

        # Keep existing prompts unchanged
        self.prompt_one_paper = """You are a research assistant analyzing a single scientific paper. Provide clear and concise answers based on the context below. Limit your response to just the specific information requested. If the answer is not found, return a "Answer for this question is not provided". If question is asked to explain something, give a brief on what is asked. Context: {context} Semantic Context: {semantic_context} Question: {question} Answer:"""

        self.prompt_multiple_papers = """You are a research assistant analyzing a scientific paper. Using only the following context, provide a clear answer to the question. Limit your response to just the specific information needed, without any additional explanation. If value is asked then return only the value with its unit. The question has two parts part 1: actual question part 2: Either String or Integer/Float once the answer is fetched, depending upon the part 2, modify the answer accordingly before returning For example: for the question metal : What are all the metals ions in the ZIF-8 compound? (String), the answer should be Zn²⁺,Cu²⁺ for the question porosity_nature : What is the porous nature of the ZIF-8 compound(just specify if microporous or mesoporous or macorporous)? (String) the answer should be either microporous or mesoporous or macorporous or Null string for the question surface_area : What is the surface area of the ZIF-8 compound? (Integer/Float) the answer should be like 1171.3 m²/g or 1867 m² g⁻¹ (if the answer is any other unit, then change it to the most IUPAC unit) for the question dimension : What is the dimension of the ZIF-8 compound (say either 2D or 3D)? (String) the answer should be either 2D or 3D or Null string for the question morphology : What is the morphology of the ZIF-8 compound? (String) the answer should be like leaf-shaped or rhombic dodecahedron etc. for the question size : What is the size of the ZIF-8 compound? (Integer/Float) the answer should be a value like 270 nm, if any other units are fetched, then change it to IUPAC unit If the answer is not there in the pdf, then return a Null string If a value with new unit is fetched, then try to convert it to the unit which is widely used for that question Technical Context from Research Paper: {context} Semantic Context: {semantic_context} Question: {question} Answer:"""

    def set_current_api_key(self):
        try:
            if not self.api_keys[self.current_key_index].strip():
                raise ValueError("Empty API key")
            genai.configure(api_key=self.api_keys[self.current_key_index])
            return True
        except Exception as e:
            st.error(f"API key {self.current_key_index + 1} is invalid or expired. Trying next key...")
            return False

    def ask_question(self, question, option=""):
        if not self.documents:
            return ""
        
        try:
            context = ""
            for text in self.texts:
                context += text.page_content + "\n"
            
            semantic_context = self.expand_semantic_context(question)
            
            if option == "Chatbot for 1 Paper":
                prompt = self.prompt_one_paper.format(
                    context=context,
                    semantic_context=semantic_context,
                    question=question
                )
            else:
                prompt = self.prompt_multiple_papers.format(
                    context=context,
                    semantic_context=semantic_context,
                    question=question
                )
            
            max_retries = len(self.api_keys) * 2  # Allow each key to be tried twice
            retry_count = 0
            
            while retry_count < max_retries:
                try:
                    if not self.set_current_api_key():
                        self.current_key_index = (self.current_key_index + 1) % len(self.api_keys)
                        retry_count += 1
                        continue

                    response = self.model.generate_content(prompt)
                    
                    if response and response.text.strip():
                        answer = response.text.strip()
                        if "does not contain" in answer.lower() or "does not give" in answer.lower():
                            return ""
                        return answer
                    
                except Exception as e:
                    error_message = str(e)
                    if "429" in error_message:  # Rate limit error
                        st.warning(f"Rate limit reached for API key {self.current_key_index + 1}. Trying next key...")
                        self.current_key_index = (self.current_key_index + 1) % len(self.api_keys)
                        time.sleep(1)
                    else:
                        st.error(f"Error with API key {self.current_key_index + 1}: {error_message}")
                        self.current_key_index = (self.current_key_index + 1) % len(self.api_keys)
                    
                    retry_count += 1
            
            st.error("All API keys have been tried and failed. Please check your API keys.")
            return ""
                
        except Exception as e:
            st.error(f"An error occurred: {str(e)}")
            return ""

def main():
    st.set_page_config(page_title="Research Chatbot", layout="wide")
    
    # Keep existing style definitions...

    st.title("Research Chatbot")
    
    # Add API key status indicator
    with st.sidebar:
        st.write("API Key Status:")
        chatbot = GeminiResearchChatbot()
        for i, key in enumerate(chatbot.api_keys):
            if key.strip():
                st.success(f"API Key {i+1}: Available")
            else:
                st.error(f"API Key {i+1}: Missing or Invalid")

    # Initialize session states
    if "questions" not in st.session_state:
        st.session_state["questions"] = []
        st.session_state["types"] = []
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []
    if "question_added" not in st.session_state:
        st.session_state.question_added = False

    option = st.selectbox("Choose an option", ["Chatbot for 1 Paper", "Compare Multiple Papers"])

    if option == "Compare Multiple Papers":
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
                
        # Show all questions
        if st.session_state["questions"]:
            st.write("### Current Questions:")
            for i, (q, t) in enumerate(zip(st.session_state["questions"], st.session_state["types"])):
                st.write(f"{i+1}. {q} ({t})")

        uploaded_files = st.file_uploader("Upload your PDF files", type="pdf", accept_multiple_files=True)

        if uploaded_files and st.session_state["questions"]:
            if st.button("Process PDFs"):
                all_results = []
                error_occurred = False
                
                with st.spinner("Processing PDFs..."):
                    progress_bar = st.progress(0)
                    status_text = st.empty()
                    
                    for i, uploaded_file in enumerate(uploaded_files):
                        try:
                            status_text.text(f"Processing {uploaded_file.name}...")
                            
                            if chatbot.load_pdf(uploaded_file):
                                results_for_pdf = {"File": uploaded_file.name}
                                
                                for question, q_type in zip(st.session_state["questions"], st.session_state["types"]):
                                    status_text.text(f"Processing {uploaded_file.name} - Question: {question}")
                                    column_title = question.split(":", 1)[0].strip() if ":" in question else question.strip()
                                    full_question = f"{question} ({q_type})"
                                    answer = chatbot.ask_question(full_question)
                                    
                                    if answer == "":
                                        status_text.warning(f"No answer found for question: {question}")
                                    
                                    results_for_pdf[column_title] = f"{answer}" if answer else "N/A"
                                
                                all_results.append(results_for_pdf)
                            else:
                                st.error(f"Failed to load PDF: {uploaded_file.name}")
                                error_occurred = True
                            
                        except Exception as e:
                            st.error(f"Error processing {uploaded_file.name}: {str(e)}")
                            error_occurred = True
                        
                        progress_bar.progress((i + 1) / len(uploaded_files))
                    
                    status_text.empty()

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
                
                if error_occurred:
                    st.warning("Some errors occurred during processing. Please check the messages above.")
                    
        elif uploaded_files and not st.session_state["questions"]:
            st.warning("Please add at least one question before processing PDFs.")
        elif not uploaded_files and st.session_state["questions"]:
            st.info("Please upload PDF files to process.")

    # Keep the single paper chatbot code unchanged...

if __name__ == "__main__":
    main()