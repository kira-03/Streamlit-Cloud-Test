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
    def __init__(self, semantics_path="Semantics.json"):
        # Get API keys from Streamlit secrets
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
        self.semantics = self.load_semantics(semantics_path)
        self.prompt_one_paper = """You are a research assistant analyzing a single scientific paper. Provide clear and concise answers based on the context below. Limit your response to just the specific information requested. If the answer is not found, return a "Answer for this question is not provided". If question is asked to explain something, give a brief on what is asked. 
            Context:
            {context}
            Semantic Context:
            {semantic_context}
            Question: {question}
            Answer:"""
        self.prompt_multiple_papers = """You are a research assistant analyzing a scientific paper. Using only the following context, provide a clear answer to the question. Limit your response to just the specific information needed, without any additional explanation. If value is asked then return only the value with its unit. 
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

            If the answer is not there in the pdf, then return a Null string

            If a value with new unit is fetched, then try to convert it to the unit which is widely used for that question


            Technical Context from Research Paper:
            {context}

            Semantic Context:
            {semantic_context}

            Question: {question}

            Answer:"""

    # Rest of the class remains exactly the same
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
                    'relations_text': ', '.join([f"{rel}: {', '.join(values)}" for rel, values in details.get('relations', {}).items()])
                }
            return processed_semantics
        except Exception as e:
            st.error(f"Error loading semantics: {str(e)}")
            return {}

    def expand_semantic_context(self, question):
        expanded_context = ""
        matched_categories = set()

        for category, details in self.semantics.items():
            matches = (
                category.lower() in question.lower() or 
                any(attr.lower() in question.lower() for attr in details['attributes']) or
                any(keyword.lower() in question.lower() for keyword in 
                    (category.split() + details['attributes'] + 
                     list(details['relations'].keys()) + 
                     [item for sublist in details['relations'].values() for item in sublist])
                ))
            
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

    def ask_question(self, question, option=""):
        if not self.documents:
            return ""

        try:
            context = ""
            for text in self.texts:
                context += text.page_content + "\n"

            semantic_context = self.expand_semantic_context(question)

            # Use appropriate prompt based on the selected option
            if option == "Chatbot for 1 Paper":
                prompt = self.prompt_one_paper.format(context=context, semantic_context=semantic_context, question=question)
            else:
                prompt = self.prompt_multiple_papers.format(context=context, semantic_context=semantic_context, question=question)

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
    st.set_page_config(page_title="Research Chatbot", layout="wide")

    st.markdown("""
        <style>
        body {
            background-color: #121212;
            color: white;
        }
        .chat-bubble {
            padding: 10px 15px;
            border-radius: 20px;
            margin: 5px 0;
            display: inline-block;
        }
        .user-bubble {
            background-color: #333333;
            color: white;
            text-align: left;
            float: left;
            clear: both;
        }
        .bot-bubble {
            background-color: #555555;
            color: white;
            text-align: right;
            float: right;
            clear: both;
        }
        .chat-container {
            max-height: 400px;
            overflow-y: auto;
            padding: 10px;
            border: 1px solid #ccc;
            border-radius: 10px;
            background: #1e1e1e;
        }
        .input-container {
            margin-top: 10px;
        }
        </style>
    """, unsafe_allow_html=True)

    # Initialize chatbot with secrets
    chatbot = GeminiResearchChatbot()

    st.title("Research Chatbot")

    option = st.selectbox("Choose an option", ["Chatbot for 1 Paper", "Compare Multiple Papers"])
    if option == "Chatbot for 1 Paper":
        uploaded_file = st.file_uploader("Upload a PDF file", type="pdf")

        if uploaded_file:
            st.write("Processing the uploaded PDF...")
            if chatbot.load_pdf(uploaded_file):
                st.success("PDF successfully processed! You can now ask questions.")

                if "chat_history" not in st.session_state:
                    st.session_state.chat_history = []

                st.write("### Chat History")
                chat_container = st.empty()

                def render_chat():
                    with chat_container.container():
                        for message in st.session_state.chat_history:
                            if message["sender"] == "user":
                                st.markdown(f"<div class='chat-bubble user-bubble'>{message['content']}</div>", unsafe_allow_html=True)
                            else:
                                st.markdown(f"<div class='chat-bubble bot-bubble'>{message['content']}</div>", unsafe_allow_html=True)

                render_chat()

                with st.container():
                    question = st.text_input("Type your question here...")
                    if st.button("Send", key="send_button") and question.strip():
                        st.session_state.chat_history.append({"sender": "user", "content": question})
                        render_chat()
                        answer = chatbot.ask_question(question, "Chatbot for 1 Paper")
                        st.session_state.chat_history.append({"sender": "bot", "content": answer})
                        render_chat()

    elif option == "Compare Multiple Papers":
        if "questions" not in st.session_state:
            st.session_state["questions"] = []
            st.session_state["types"] = []

        new_question = st.text_input("Enter a question:")
        answer_type = st.selectbox("Select the answer type for the question:", ["String", "Integer/Float"])

        if st.button("Add Question"):
            if new_question.strip():
                st.session_state["questions"].append(new_question.strip())
                st.session_state["types"].append(answer_type)
                st.success(f"Question added: {new_question} ({answer_type})")
            else:
                st.warning("Question cannot be empty.")

        uploaded_files = st.file_uploader("Upload your PDF files", type="pdf", accept_multiple_files=True)

        if uploaded_files:
            if "results" not in st.session_state:
                all_results = []

                for uploaded_file in uploaded_files:
                    if chatbot.load_pdf(uploaded_file):
                        st.success(f"PDF '{uploaded_file.name}' loaded successfully!")

                        results_for_pdf = {"File": uploaded_file.name}
                        for question, q_type in zip(st.session_state["questions"], st.session_state["types"]):
                            column_title = question.split(":", 1)[0].strip() if ":" in question else question.strip()
                            answer = chatbot.ask_question(question)
                            results_for_pdf[column_title] = f"{answer}" if answer else "N/A"

                        all_results.append(results_for_pdf)

                st.session_state["results"] = pd.DataFrame(all_results)

            df = st.session_state["results"]
            st.write(df)

            if not df.empty:
                st.write("### Filter Results")
                original_df = st.session_state["results"]

                for column, q_type, question in zip([col for col in df.columns if col.lower() != "file"], 
                                                    st.session_state["types"], 
                                                    st.session_state["questions"]):
                    df[column] = df[column].astype(str)

                    if q_type == "Integer/Float":
                        pattern = r"^\s*\d+(\.\d+)?(\s*±\s*\d+(\.\d+)?)?\s*[a-zA-Z²⁻¹/]*.*$"
                        matches_format = df[column].str.match(pattern).fillna(False)

                        df.loc[~matches_format, column] = None

                        if matches_format.any():
                            numeric_values = df[column].str.extract(r"([0-9]+(?:\.[0-9]+)?)")[0]
                            numeric_values = pd.to_numeric(numeric_values, errors='coerce')

                            if numeric_values.notnull().any():
                                min_val, max_val = numeric_values.min(), numeric_values.max()
                                if min_val == max_val:
                                    max_val = min_val + 1

                                lower_limit, upper_limit = st.slider(
                                    f"Filter for {column}:",
                                    min_value=float(min_val),
                                    max_value=float(max_val),
                                    value=(float(min_val), float(max_val)),
                                    step=0.1,
                                )

                                df = df[
                                    numeric_values.between(lower_limit, upper_limit, inclusive="both") | 
                                    df[column].isnull()
                                ]

                    else:
                        filter_value = st.text_input(f"Filter for {column}:")
                        if filter_value.strip():
                            df = df[
                                df[column].str.contains(filter_value, case=False, na=False) | 
                                df[column].isnull()
                            ]

                st.write("### Filtered Results")
                st.dataframe(df)

if __name__ == "__main__":
    main()