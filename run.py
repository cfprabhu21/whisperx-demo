from configparser import ConfigParser
from pathlib import Path
import os

from service import WhisperX

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
            diarized_transcript = wx.diarize(transcribe)
            diarized_transcript_text_file_name = file_base_name + 'diarized_transcript.txt'
            with open(Path(output_path, diarized_transcript_text_file_name), 'w') as f:
                for segment in diarized_transcript["segments"]:
                    speaker_label = segment.get("speaker", "Unknown")
                    text = segment.get("text", "")
                    # summery=summarize_text(text)
                    f.write(f"Speaker {speaker_label}: {text}\n")

            diarized_transcript['language'] = 'en'
            wx.generate_vtt(diarized_transcript, output_path)
            wx.summarize(Path(output_path, diarized_transcript_text_file_name))


def sorted_dir(folder):
    def getmtime(name):
        path = os.path.join(folder, name)
        return os.stat(path).st_mtime

    return sorted(os.listdir(folder), key=getmtime, reverse=False)



if __name__ == '__main__':
    run()