import PyPDF2
from youtube_transcript_api import YouTubeTranscriptApi
from langchain_community.document_loaders import UnstructuredURLLoader



class DataExtractor:
    def extract_text_from_pdf(pdf_path):
        with open(pdf_path, 'rb') as file:
            reader = PyPDF2.PdfReader(file)
            text = ""
            for page_num in range(len(reader.pages)):
                page = reader.pages[page_num]
                text += page.extract_text()
        return text


class VideoExtractor:
    def extract_text_from_video(video_id):
        transcript = YouTubeTranscriptApi.get_transcript(video_id)
        text = ""
        for line in transcript:
            text += line['text'] + " "
        return text
    

class UnstructuredURLExtractor:
    def extract_text_from_url(url):
        loader = UnstructuredURLLoader(url)
        text = loader.load()
        return text
    


