import os
import typing as t

from pathlib import Path
from pprint import pprint

import whisperx
from whisperx.utils import get_writer
import os
from configparser import ConfigParser

from log_text_summarization import LongTextSummarizationPipeline

LANGUAGE_CODE = "en"


class WhisperX:
    """
    Expectation by the weekend:
    1. Read the file path and get the audio or video file.Create the json file with the following object scoring, original file(which we got on back of whisper x), summary, transcription, vtt file
    2. save the json file and the transcription object in a word document in an output file path
    3. the input file path and output file path should be in config file so that we can change
    We need this asap
    """
    def __init__(self, audio_file: Path):
        import torch
        import whisperx

        self.audio_file = audio_file

        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.batch_size = 16  #reduce if low on GPU memory
        compute_type = "float16" if torch.cuda.is_available() else "float32whisper"
        self.model = whisperx.load_model("large-v2", self.device, compute_type=compute_type, language=LANGUAGE_CODE)
        self.model_a, self.metadata = whisperx.load_align_model(language_code=LANGUAGE_CODE, device=self.device)
        self.writer_options = {
            "max_line_width": None,  # Adjust these options based on your needs and the writers' requirements
            "max_line_count": None,
            "highlight_words": True,
            # Add any other options required by the writers for different formats
        }

    def transcribe(self) -> t.Dict:
        audio = whisperx.load_audio(self.audio_file)
        result = self.model.transcribe(audio, batch_size=self.batch_size, print_progress=True, combined_progress=True)
        pprint(result["segments"])  # Print transcription segments before alignment
        result = whisperx.align(result["segments"], self.model_a, self.metadata, audio, self.device,
                                return_char_alignments=False)
        print(result["segments"])  # Print aligned transcription segments
        return result

    def diarize(self, result) -> t.Dict:
        diarize_model = whisperx.DiarizationPipeline(use_auth_token="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", device=self.device)
        diarize_segments = diarize_model(self.audio_file, min_speakers=1, max_speakers=5)
        result = whisperx.assign_word_speakers(diarize_segments, result)
        print(result["segments"])  # Print segments with assigned speaker IDs
        return result

    def generate_vtt(self, result, output_dir):
        if not os.path.exists(output_dir):
            os.makedirs(output_dir, exist_ok=True)
        writer = whisperx.utils.get_writer("vtt", output_dir)  # Get the writer for the current format
        writer(result, self.audio_file, self.writer_options)
        print("Transcription results saved in all specified formats.")

    def summarize(self, transcript_file) -> t.Dict:
        summarizer = LongTextSummarizationPipeline(model_id="facebook/bart-large-cnn")
        with open(transcript_file, "r", encoding='utf-8') as file:
            # Read the entire content of the file
            transcript = file.read()
        summary_text = summarizer.summarize(transcript)
        pprint(summary_text)
        return summary_text

