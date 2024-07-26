# Apple Vision Pro Q/A Chatbot

This project is an LLM-based Retrieval-Augmented Generation (RAG) Chatbot Application designed to answer questions about the Apple Vision Pro. The application leverages a vector database to store embeddings of relevant documents and retrieves contextual information to provide accurate responses using the Gemini API.

## Features

- Extract data from PDFs, websites, and YouTube videos.
- Store extracted data in a vector database (FAISS).
- Retrieve relevant information from the vector database.
- Generate responses using the Gemini API.
- Interactive chatbot interface built with Streamlit.

## Setup and Installation

### Prerequisites

- Python 3.8 or higher
- Virtual environment (recommended)
- Required Python packages listed in `requirements.txt`

### Environment Variables

Create a `.env` file in the project root directory with the following content:

```env
GEMINI_API_KEY=<your_gemini_api_key>
```

### Installation Steps

1. **Clone the Repository**

    ```bash
    git clone https://github.com/Vishesh-30/DeepSolv-assignment
    cd DeepSolv-assignment
    ```

2. **Create a Virtual Environment and Activate it**

    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
    ```

3. **Install the Required Packages**

    ```bash
    pip install -r requirements.txt
    ```

4. **Run the Application**

    ```bash
    Python app.py
    ```
    and
    ```bash
    streamlit run chatbot.py
    ```

## Project Structure

```plaintext
DeepSolv-assignment/
├── src/
│   ├── data_extraction.py      # Data extraction from PDFs, websites, YouTube
│   ├── embedding_generator.py  # Generates embeddings using SentenceTransformer
│   ├── vector_store.py         # Manages FAISS vector store operations
├── app.py                      # Main python file
├── chatbot.py                  # Chatbot application using Gemini API
├── requirements.txt            # Required Python packages
├── .env                        # Environment variables
└── README.md                   # This README file
```

## Usage

1. **Data Extraction**

   The `data_extraction.py` script extracts data from various sources and stores it in a format suitable for embedding generation.

2. **Generate Embeddings**

   The `embedding_generator.py` script generates embeddings from the extracted text data using SentenceTransformer.

3. **Manage Vector Store**

   The `vector_store.py` script handles the creation, loading, and searching of the FAISS index.

4. **Run the Chatbot**

   The `app.py` script runs the Streamlit application, providing an interactive interface for users to ask questions and receive responses.

## Example Usage

1. Start the application:
    ```bash
    python app.py
    ```
    and

    ```bash
    streamlit run app.py
    ```

2. Open the provided URL in your web browser to interact with the chatbot.

3. Enter a question about the Apple Vision Pro in the input box at the bottom of the screen.

4. The chatbot retrieves relevant information from the vector database and generates a response using the Gemini API.

## Customization

- **Model Configuration**: You can customize the embedding model used in `embedding_generator.py` by changing the `model_name` parameter.
- **Data Sources**: You can add more data sources or modify the existing ones in `data_extraction.py`.
- **Front-End Customization**: Modify the Streamlit layout and appearance in `app.py` using custom CSS and Streamlit widgets.

## Troubleshooting

- Ensure that all environment variables are correctly set in the `.env` file.
- Make sure the FAISS index file (`faiss_index.bin`) is accessible and correctly loaded.
- Check for any missing dependencies and install them using `pip install -r requirements.txt`.

## Contributing

Contributions are welcome! Please fork the repository and submit a pull request for any enhancements or bug fixes.

## License

This project is licensed under the MIT License. See the LICENSE file for details.

---

Feel free to adjust the content as per your project's specifics and additional details you might want to include.
