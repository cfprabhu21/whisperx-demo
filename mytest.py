import torch
from transformers import AutoTokenizer, pipeline
from pprint import pprint

model_id = "facebook/bart-large-cnn"
device = "cuda" if torch.cuda.is_available() else "cpu"
tokenizer = AutoTokenizer.from_pretrained(model_id)
model = pipeline(
                "summarization",
                model=model_id,
                device=-1,
            )
model_max_length = tokenizer.model_max_length
print(model_max_length)

text = """
    'Social psychologists say there are six ways of thinking about time, which '
 'are called personal time zones. People living in the past negative time zone '
 'are absorbed by earlier times. People classified as future active are the '
 'planners and go-getters. Children need to be aware of how other people think '
 'about time. Time perspectives make a big difference in how we value and use '
 'our time. Seeing conflicts as differences in time perspective, rather than '
 'character, can facilitate more effective cooperation.')
"""
with open("diarized_transcript.txt", "r", encoding='utf-8') as file:
    # Read the entire content of the file
    transcript = file.read()

tokens = tokenizer.encode(transcript, truncation=False, return_tensors="pt")
# pprint(tokens.size(1))
# print(model(text, max_length=130, min_length=30, do_sample=False))

tokens1 = tokenizer.encode(text)
# pprint(tokens1)

# chunk_size = model_max_length // 2  # Use half the max length for overlap
# overlap = chunk_size // 5  # Define overlap size
# tokens = tokenizer.encode(text, truncation=False)
# chunks = []
#
# for i in range(0, len(tokens), chunk_size - overlap):
#     end = min(i + chunk_size, len(tokens))
#     chunk_tokens = tokens[i:end]
#     chunk_text = tokenizer.decode(chunk_tokens)
#     pprint(chunk_text)
#     summary = model(chunk_text, max_length=model_max_length, min_length=30, do_sample=False)
#     chunks.append(summary[0]['summary_text'])
#
# # Combine the summarized chunks
# combined_summary = " ".join(chunks)
# pprint(combined_summary)

tokens = tokenizer.encode(text)
chunk_size = model_max_length - 50
chunk_start = 0
chunks = []
while chunk_start < len(tokens):
    chunk_end = min(chunk_start + chunk_size, len(tokens))
    chunk = tokenizer.decode(
        tokens[chunk_start:chunk_end],
        skip_special_tokens=True,
        clean_up_tokenization_spaces=True,
    )
    chunks.append(chunk)
    chunk_start += chunk_size
pprint(chunks)