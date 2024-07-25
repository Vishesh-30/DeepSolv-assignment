from src.data_extraction import DataExtractor, VideoExtractor
from src.embedding_generator import EmbeddingGenerator
from src.vector_store import VectorStore
from src.data_extraction import UnstructuredURLExtractor
from src.embedding_generator import RecursiveTextSplitter

import numpy as np




# Step 1: Extract text from PDF
pdf_path = "Apple_Vision_Pro_Privacy_Overview.pdf"
extracted_text = DataExtractor.extract_text_from_pdf(pdf_path)

# Extract text from Subtitles/Transcript
video_id = "TX9qSaGXFyg"
transcript = VideoExtractor.extract_text_from_video(video_id)
# print(transcript)

# Extract text from URL
url = "https://www.apple.com/apple-vision-pro/"
url_text = UnstructuredURLExtractor.extract_text_from_url([url])
# print(url_text[0].page_content)

extracted_text = extracted_text + transcript + url_text[0].page_content
# print(extracted_text)

# Split extracted text into chunks if needed (FAISS works better with shorter texts)
text_chunks = RecursiveTextSplitter.split_text(extracted_text)
# print(text_chunks)

# Step 2: Generate embeddings
# Extract text content from Document objects
text_chunks = [chunk.page_content for chunk in text_chunks]

# Generate embeddings for text content
embedding_generator = EmbeddingGenerator()
embeddings = embedding_generator.generate_embeddings(text_chunks)

# Step 3: Store embeddings in FAISS
if embeddings.size > 0:
    dimension = embeddings.shape[1]
    vector_store = VectorStore(dimension)
    vector_store.add_embeddings(embeddings)

    # Save the FAISS index for later use
    index_path = "faiss_index.bin"
    vector_store.save_index(index_path)
    print(f"FAISS index saved to {index_path}")

    # Save the text chunks to a file (optional, for retrieval in chatbot)
    chunks_path = "text_chunks.npy"
    np.save(chunks_path, text_chunks)
    print(f"Text chunks saved to {chunks_path}")

    print("Embeddings have been successfully stored in FAISS index.")



