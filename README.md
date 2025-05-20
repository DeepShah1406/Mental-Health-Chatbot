# **Mental Health Chatbot**

**Final Year Project**
_A chatbot for mental health support using Retrieval-Augmented Generation (RAG) and trusted data sources._

---

## **Description**

The **Mental Health Chatbot** is built to provide real-time, reliable mental health-related information and support. The chatbot uses **Retrieval-Augmented Generation (RAG)** techniques to fetch data from vectorized mental health content sourced from authoritative institutions such as the **World Health Organization (WHO)** and the **Ministry of Health and Family Welfare, Government of India**.

Sensitive keywords like "self-harm" and "harm" are detected by the system, triggering responses with emergency helpline numbers, ensuring users get immediate help in distressing situations.

The bot is designed to support users by offering not only information but also crucial helplines when they might need it most.

---

## **Features**

- **Accurate Mental Health Information**: Fetches contextually relevant information from authoritative sources such as WHO and the Ministry of Health and Family Welfare, India.
- **Sensitive Topic Detection**: Automatically detects distressing keywords like "self-harm" or "harm" and responds with helpline numbers for immediate assistance.
- **Retrieval-Augmented Generation (RAG)**: Combines retrieval of relevant data from vector databases with generative responses for detailed and contextual answers.
- **User-Friendly Interface**: Developed with **Streamlit** to provide a smooth and intuitive interface for real-time interaction.
- **Real-Time Data Processing**: Uses **LangChain** for natural language processing and real-time querying of vectorized mental health data.
- **Text and PDF Processing**: Processes downloaded PDFs containing publicly available mental health data, enabling efficient querying of static resources.
- **Static Data Source**: The current data is based on static PDF documents but will be updated regularly in the future.

---

## **Tech Stack**

- **Python 3.13.3**
- **Streamlit 1.38.0**: For developing the chatbot's user interface.
- **LangChain**: For building conversational AI and handling text data processing.

  - **langchain-community==0.2.16**
  - **langchain-text-splitters==0.2.4**
  - **langchain-chroma==0.1.3**: For managing vectorized mental health data.
  - **langchain-huggingface==0.0.3**: For integrating HuggingFace models.
  - **langchain-groq==0.1.9**

- **HuggingFaceEmbeddings**: For embedding mental health documents and improving data retrieval.
- **PyPDF2**: For reading and extracting content from mental health-related PDFs.
- **NLTK 3.8.1**: For basic text processing and tokenization.
- **pytesseract\~=0.3.13**: For OCR if processing image-based PDFs or scanned documents.
- **python-magic-bin==0.4.14**: For identifying file types when processing documents.

**LLM Model:**

- **ChatGroq** (Model: `llama-3.3-70b-versatile` with temperature set to 0): A large language model used to generate responses based on user input.

---

## **How to Run Locally**

### **Prerequisites**

1. **Python 3.13.3** or higher.
2. Install required dependencies:

   ```bash
   pip install streamlit==1.38.0 langchain-community==0.2.16 langchain-text-splitters==0.2.4 langchain-chroma==0.1.3 langchain-huggingface==0.0.3 nltk==3.8.1 pytesseract==0.3.13 PyPDF2==3.0.1 python-magic-bin==0.4.14
   ```

### **Running the Chatbot**

1. Clone this repository:

   ```bash
   git clone https://github.com/yourusername/mental-health-chatbot.git
   cd mental-health-chatbot
   ```

2. Ensure the **mental health PDFs** are placed in the appropriate folder (or point the script to the location).
3. Start the Streamlit app:

   ```bash
   streamlit run app.py
   ```

4. The app will open in your browser where you can interact with the chatbot.

---

## **User Interaction**

- **Input**: Users can type their queries directly into the Streamlit interface.
- **Output**: The chatbot processes the query, retrieves relevant information from the vector database, and displays context-aware responses.
- **Sensitive Content**: If the system detects sensitive terms like "self-harm" or "harm," it will immediately provide helpline numbers and offer emergency resources.

---

## **Data Sources**

- **Mental Health PDFs**: The initial data for this chatbot is based on publicly available PDFs downloaded from trusted sources like WHO and the Ministry of Health and Family Welfare. These documents contain important mental health guidelines and helpline information.
- **Chroma Vector Database**: The data is vectorized and stored in the **Chroma** vector database, allowing for efficient querying and retrieval.

---

## **Future Improvements**

- **Dynamic Data Updates**: Currently, the data is static, but future iterations will implement a mechanism to update the mental health information regularly to stay current with new guidelines, data, and helplines.
- **Expanded Data Sources**: We plan to expand the data sources by integrating other publicly available mental health resources, including articles, online counseling services, and data from mental health professionals.
- **Personalized Responses**: Future versions may allow for a personalized user experience, providing suggestions based on a user’s emotional state or past interactions.
- **Integration with Mental Health Resources**: We aim to provide links to clinics, doctors, online counseling, and more.

---

## **Contributing**

If you’d like to contribute to the project, feel free to fork this repository, make your changes, and submit a pull request.

---

## **License**

This project is licensed under the MIT License - see the [LICENSE](https://github.com/DeepShah1406/Mental-Health-Chatbot/blob/master/LICENSE) file for details.

---

## Contact

- 📧 Email: [shahdeep1406@gmail.com](mailto:shahdeep1406@gmail.com)
- 🐙 GitHub: [DeepShah1406](https://github.com/DeepShah1406)
- 💼 LinkedIn: [deepshah1406](https://www.linkedin.com/in/deepshah1406)

---

## **Acknowledgments**

- **LangChain**: For building robust chains of logic for querying and text processing.
- **Chroma**: For efficiently managing and querying vectorized data.
- **WHO** and **Ministry of Health and Family Welfare, Government of India**: For providing the essential mental health guidelines.
