# %%
import os
import time
import pandas as pd
import requests
import json

OPENAI_TOKEN = 'paste your token here'
# %%
files = os.listdir('texts')
# %%
def send_request(text_to_send):
    url = "https://api.openai.com/v1/chat/completions"
    payload = json.dumps({
    "model": "gpt-3.5-turbo",
    "messages": [
        {
        "role": "user",
        "content": text_to_send
        }
    ],
    "temperature": 1,
    "top_p": 1,
    "n": 1,
    "stream": False,
    "max_tokens": 750,
    "presence_penalty": 0,
    "frequency_penalty": 0
    })
    headers = {
    'Content-Type': 'application/json',
    'Accept': 'application/json',
    'Authorization': OPENAI_TOKEN
    }

    return requests.request("POST", url, headers=headers, data=payload)
#%%

# %%
results = []
for file in files:
    with open(os.path.join('texts', file), 'r') as f:
        text = f.read()
    print(file, len(text.split()))
    window_len = 150
    if len(text.split()) < window_len:
        print(f'Insufficient words in {file} - {len(text.split())}')
    for i in range(0, len(text.split())-window_len, window_len):
        #print(" ".join(text.split()[i:i+window_len]))
        line = " ".join(text.split()[i:i+window_len])

        query = "Rephrase text to avoid plagiarism: " + line
        while True:
            time_0 = time.time()
            response = send_request(line)  
            res = json.loads(response.text)
            if response.status_code != 200:
                print(res)
                continue
            response_text = res['choices'][0]['message']['content'].replace('\n', ' ').replace('  ', ' ')
            delta_t = time.time() - time_0
            results.append([file, line, response_text, res['usage'], delta_t])
            break
# %%
import pickle
# %%
with open('all.pkl', 'wb') as f:
    pickle.dump(results, f)
# %%
with open('all.pkl', 'rb') as f:
    results = pickle.load(f)

# %%
df = pd.DataFrame(results, columns=['file', 'original', 'rephrased', 'usage'])
# %%
from datasets import Dataset
# %%
def gen():
    for row in results:
        yield {
            "file": row[0],
            "orig": row[1],
            "response": row[2],
            "usage": row[3],
            "delta_t": row[4]
        }
# %%
ds = Dataset.from_generator(gen)

def mapper(batch):
    print(batch)
    return {
        "orig_len": [ len(string) for string in batch['orig']],
        "response_len": [ len(string) for string in batch['response']],
        "orig_words": [ len(string.split()) for string in batch['orig']],
        "response_words": [ len(string.split()) for string in batch['response']],
    }

ds2 = ds.map(function=mapper, batched=True)
# pandas again
df = pd.DataFrame(
    ds2
)

# file = None # this is beracause I f**d up before, now there issue was corrected
# window_len = 150
# for idx in range(df.shape[0]):
#     if df.iloc[idx]['file'] != file:
#         i = 0
#         file = df.iloc[idx]['file']
#         text = df.iloc[idx]['orig']

#     df.loc[idx, 'orig'] = " ".join(text.split()[i:i+window_len])
#     i+=window_len

# %%
df1 = df[['file', 'orig']]

# %%
df1.loc[:, 'label'] = 'orig'
# %%
df2 = df[['file', 'response']]
df2.loc[:, 'label'] = 'gpt'
# %%
df1 = df1.rename(columns={'orig': 'text'})
df2 = df2.rename(columns={'response': 'text'})
# %%
df3 = pd.concat([df1, df2])
# %%
df3.to_csv('df.csv')