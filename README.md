# Training part of GPT generated text detection in Bachelor's theses

## Abstract
The increasing popularity of chatbots, such as
ChatGPT, has led to a growing concern regarding
the accurate identification of AI-generated
content. AI-written text is being used more and
more frequently in everyday life, which opens
up possibilities for potential misuse. This issue
becomes even more critical when it comes
to academical theses, as existing AI detection
tools do not always reliably identify AIgenerated
text. This project aims to address
this problem by creating a dataset of thesis
works from previous students in a specific curriculum.
These theses will be rephrased using
the GPT-3.5 turbo model, and the transformerbased
model will be fine-tuned to determine
whether the text was written by humans or by
ChatGPT. Additionally, an explainability technique
will be applied to gain insight into the
words and semantic structures that influence
the model’s decision-making process.

## The workflow

Firstly, the theses of interest must be downloaded, unfortunately selenium code was iterated interactively and I don't have neither saved python script nor jupyter notebook because machine crashed.

However, after theses are saved in pdf format in `data` folder, the following steps can be followed:
1) First run `pdf_parser.py` to parse pdfs into a batch of .txt files
2) Then run `openAI.py` to get responses from GPT-3.5 turbo. Provide your API key in the code.
3) After dataset was created the model can then trained using huggingface trainer. Run `train.py`