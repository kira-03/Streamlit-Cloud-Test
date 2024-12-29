import streamlit as st
import google.generativeai as genai
import json
import pandas as pd
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
import tempfile
import os
import time
from dotenv import load_dotenv
import os

# Load environment variables from .env file
load_dotenv()

class GeminiResearchChatbot:
    def __init__(self, semantics_path="Semantics.json"):
        # Load API keys from environment variables
        self.api_keys = [
            os.getenv("GEMINI_API_KEY_1"),
            os.getenv("GEMINI_API_KEY_2"),
            os.getenv("GEMINI_API_KEY_3"),
            os.getenv("GEMINI_API_KEY_4"),
            os.getenv("GEMINI_API_KEY_5"),
            os.getenv("GEMINI_API_KEY_6")
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
        self.semantics = self.load_semantics(semantics_path)

    def set_current_api_key(self):
        genai.configure(api_key=self.api_keys[self.current_key_index])

    def load_semantics(self, path):
        try:
            with open(path, "r") as file:
                semantics = json.load(file)
            
            # Preprocess semantics for easier lookup
            processed_semantics = {}
            for category, details in semantics.items():
                processed_semantics[category] = {
                    'attributes': details.get('attributes', []),
                    'attributes_text': ', '.join(details.get('attributes', [])),
                    'relations': details.get('relations', {}),
                    'relations_text': ', '.join([ 
                        f"{rel}: {', '.join(values)}" 
                        for rel, values in details.get('relations', {}).items()
                    ])
                }
            return processed_semantics
        except Exception as e:
            st.error(f"Error loading semantics: {str(e)}")
            return {}

    def expand_semantic_context(self, question):
        expanded_context = ""
        matched_categories = set()

        # Find matching semantic categories more comprehensively
        for category, details in self.semantics.items():
            # Broader matching criteria
            matches = (
                category.lower() in question.lower() or 
                any(attr.lower() in question.lower() for attr in details['attributes']) or
                # Add more flexible matching conditions
                any(keyword.lower() in question.lower() for keyword in 
                    (category.split() + details['attributes'] + 
                     list(details['relations'].keys()) + 
                     [item for sublist in details['relations'].values() for item in sublist])
                ))
            
            if matches:
                matched_categories.add(category)
        
        # If no direct matches, include all semantics for broader context
        if not matched_categories:
            matched_categories = set(self.semantics.keys())

        # Generate expanded context for matched categories
        for category in matched_categories:
            details = self.semantics[category]
            expanded_context += f"\n--- {category} Semantic Context ---\n"
            expanded_context += f"Attributes: {details['attributes_text']}\n"
            expanded_context += f"Relations: {details['relations_text']}\n"

        return expanded_context

    def load_pdf(self, pdf_file):
        try:
            with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as temp_file:
                temp_file.write(pdf_file.read())
                temp_file_path = temp_file.name

            loader = PyPDFLoader(temp_file_path)
            self.documents = loader.load()

            if not self.documents:
                st.warning("No content found in PDF")
                return False

            self.texts = self.text_splitter.split_documents(self.documents)

            if not self.texts:
                st.warning("Failed to split document into chunks")
                return False

            return True

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
            # Gather context from PDF
            context = ""
            for text in self.texts:
                context += text.page_content + "\n"

            # Expand context with semantic information
            semantic_context = self.expand_semantic_context(question)
            expanded_context = context + "\n" + semantic_context

            prompt = f"""You are a research assistant analyzing a scientific paper. Using only the following context, provide a clear answer to the question. Limit your response to just the specific information needed, without any additional explanation. If value is asked then return only the value with its unit. 
            The question has two parts
            part 1: actual question
            part 2: Either String or Integer/Float
            once the answer is fetched, depending upon the part 2, modify the answer accoringly before returning
            For example: 
            for the question 
            metal : What are all the metals ions in the ZIF-8 compound? (String),
            the answer should be Zn²⁺,Cu²⁺
            for the question 
            porosity_nature : What is the porous nature of the ZIF-8 compound(just specify if microporous or mesoporous or macorporous)? (String)
            the answer should be either microporous or mesoporous or macorporous or Null string
            for the question
            surface_area : What is the surface area of the ZIF-8 compound? (Integer/Float)
            the answer should be like 1171.3 m²/g or 1867 m² g⁻¹ (if the answer is anyother unit, then change it to the most IUPAC unit)
            for the question 
            dimension : What is the dimension of the ZIF-8 compound (say either 2D or 3D)? (String)
            the answer should be either 2D or 3D or Null string
            for the question
            morphology : What is the morphology of the ZIF-8 compound? (String)
            the answer should be like leaf-shaped or rhombic dodecahedron etc. 
            for the question 
            size : What is the size of the ZIF-8 compound? (Integer/Float)
            the answer should be a value like 270 nm, if anyother units is fetched, then change it to IUPAC unit

            If a value with new unit is fetched, then try to convert it to the unit which is widely used for that question


            Technical Context from Research Paper:
            {context}

            Semantic Context:
            {semantic_context}

            Question: {question}

            Answer:"""

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
    chatbot = GeminiResearchChatbot()

    st.title("MOF Insight: A Specialized QA Model for Metal-Organic Framework Literature")

    if "questions" not in st.session_state:
        st.session_state["questions"] = []
        st.session_state["types"] = []

    new_question = st.text_input("Enter a question:")
    answer_type = st.selectbox("Select the answer type for the question:", ["String", "Integer/Float"])

    if st.button("Add Question"):
        if new_question.strip():
            st.session_state["questions"].append(new_question)
            st.session_state["types"].append(answer_type)

    if len(st.session_state["questions"]) > 0:
        questions_df = pd.DataFrame({
            "Questions": st.session_state["questions"],
            "Answer Types": st.session_state["types"]
        })

        st.write("Questions in the Queue:")
        st.write(questions_df)

    uploaded_file = st.file_uploader("Upload your Research Paper (PDF)", type=["pdf"])

    if uploaded_file:
        if chatbot.load_pdf(uploaded_file):
            st.success("PDF uploaded and ready for processing.")
        else:
            st.error("There was an error with your PDF.")

    if st.button("Ask"):
        question = new_question.strip()
        answer_type = answer_type.strip()
        if question:
            answer = chatbot.ask_question(question)
            if answer:
                st.write(f"Answer: {answer}")
            else:
                st.write("Sorry, I couldn't retrieve an answer.")
        else:
            st.write("Please enter a question.")

if __name__ == "__main__":
    main()
