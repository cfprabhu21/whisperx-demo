import json
import os


class ConfidenceScore:
    def __init__(self, json_file):
        if not os.path.exists(json_file):
            raise Exception("File not found")
        file_info = os.path.splitext(json_file)
        file_name = file_info[0]
        file_extension = file_info[1]
        if not file_extension == ".json":
            raise Exception("File is not a json file")
        try:
            f = open(json_file)
            self.data = json.load(f)
        except Exception as e:
            raise Exception(e)

    def get_confidence_score(self):
        scores = []
        for segment in self.data.get('segments', []):
            for word in segment.get('words', []):
                scores.append(word.get('score'))
        if len(scores) == 0:
            return 0
        total_confidence = sum(scores)
        average_confidence_score = total_confidence / len(scores)
        return average_confidence_score



