from configparser import ConfigParser
from pathlib import Path
import os
import json
from docx import Document

from service import WhisperX
from confidence_score import ConfidenceScore


def run():
    BASE_DIR = Path(__file__).parent.resolve()
    file_ = Path(BASE_DIR, 'config.ini')
    config = ConfigParser()
    config.read(file_)
    source_path = config.get("filepath", "source_path")
    output_path = config.get("filepath", "output_path")
    files = sorted_dir(source_path)
    print(len(files))
    if len(files):
        for index in range(1):
            file = Path(source_path, files[index])
            file_name = file.name
            file_base_name = file_name.split('.')[0]

            wx = WhisperX(file)
            transcribe = wx.transcribe()

            # create the transcripe jon file
            transcript_json_file_name = file_base_name + '_transcript_json.json'
            transcript_json_file = convert_path_to_string(output_path, transcript_json_file_name)
            with open(transcript_json_file, 'w',
                      encoding='utf-8') as json_file:
                json.dump(transcribe, json_file, ensure_ascii=False, indent=4)

            confidence_score = ConfidenceScore(transcript_json_file).get_confidence_score()
            print(f"confidence score is {confidence_score}")

            diarized_transcript = wx.diarize(transcribe)
            diarized_transcript_text_file_name = file_base_name + 'diarized_transcript.txt'

            document = Document()
            diarized_transcript_word_file_name = file_base_name + 'diarized_transcript_word.docx'

            with open(convert_path_to_string(output_path, diarized_transcript_text_file_name), 'w') as f:
                for segment in diarized_transcript["segments"]:
                    speaker_label = segment.get("speaker", "Unknown")
                    text = segment.get("text", "")
                    # summery=summarize_text(text)
                    f.write(f"Speaker {speaker_label}: {text}\n")
                    document.add_paragraph(f"Speaker {speaker_label}: {text}\n")
            document.save(convert_path_to_string(output_path, diarized_transcript_word_file_name))

            diarized_transcript['language'] = 'en'
            wx.generate_vtt(diarized_transcript, output_path)
            summary = wx.summarize(Path(output_path, diarized_transcript_text_file_name))

            summary_text_file_name = file_base_name + '_summary.txt'
            with open(Path(output_path, summary_text_file_name), 'w') as summary_file:
                summary_file.write(json.dumps(summary))


def sorted_dir(folder):
    def getmtime(name):
        path = os.path.join(folder, name)
        return os.stat(path).st_mtime

    return sorted(os.listdir(folder), key=getmtime, reverse=False)


def convert_path_to_string(output_path, file_name):
    return str(Path(output_path, file_name))


if __name__ == '__main__':
    run()
