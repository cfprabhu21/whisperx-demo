import torch
from transformers import AutoTokenizer, pipeline
import whisperx


class AudioProcessor:
    def __init__(self, whisper_model_name="base", summarization_model_name="facebook/bart-large-cnn"):
        # Initialize Whisper model
        self.whisper_model = whisperx.load_model(whisper_model_name)

        # Initialize BART model for summarization
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.tokenizer = AutoTokenizer.from_pretrained(summarization_model_name)
        self.summarizer = pipeline(
            "summarization",
            model=summarization_model_name,
            device=0 if self.device == "cuda" else -1
        )
        self.model_max_length = self.tokenizer.model_max_length

    def transcribe_audio(self, file_path):
        # Transcribe audio file
        result = self.whisper_model.transcribe(file_path)
        return result['text']

    def summarize_text(self, text):
        # Tokenize the text
        tokens = self.tokenizer.encode(text, truncation=False, return_tensors="pt")
        token_length = tokens.size(1)

        # Check if token length exceeds model's maximum length
        if token_length > self.model_max_length:
            print("Text is too long. Splitting into smaller chunks.")
            return self._summarize_in_chunks(text)
        else:
            summary = self.summarizer(text, max_length=self.model_max_length, min_length=30, do_sample=False)
            return summary[0]['summary_text']

    def _summarize_in_chunks(self, text):
        # Split the text into smaller chunks
        chunk_size = self.model_max_length // 2  # Use half the max length for overlap
        overlap = chunk_size // 5  # Define overlap size
        tokens = self.tokenizer.encode(text, truncation=False)
        chunks = []

        for i in range(0, len(tokens), chunk_size - overlap):
            end = min(i + chunk_size, len(tokens))
            chunk_tokens = tokens[i:end]
            chunk_text = self.tokenizer.decode(chunk_tokens)
            summary = self.summarizer(chunk_text, max_length=self.model_max_length, min_length=30, do_sample=False)
            chunks.append(summary[0]['summary_text'])

        # Combine the summarized chunks
        combined_summary = " ".join(chunks)
        return combined_summary


if __name__ == "__main__":
    audio_file_path = "path_to_your_audio_file.wav"  # Replace with your audio file path
    processor = AudioProcessor()

    # Transcribe the audio
    transcript = processor.transcribe_audio(audio_file_path)
    print("Transcript:")
    print(transcript)

    # Summarize the transcript
    summary = processor.summarize_text(transcript)
    if summary:
        print("Summary:")
        print(summary)
