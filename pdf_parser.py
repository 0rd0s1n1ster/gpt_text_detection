# %%
import os
import time
import PyPDF2
import pandas as pd
# %%
PDFS_FOLDER = 'data'
files = os.listdir(PDFS_FOLDER)
# %%
# %%
saving = False
# %% 
def visitor(text, cm, tm, fontDict, fontsize):
    global saving
    y = cm[5]
    if (fontsize > 12 or fontsize==1):
        # if text contains introduction savign true
        if 'ntroduction' in text.lower():
            print(f'found introduction, fontsize {fontsize}, y - {cm[5]}, {tm[5]}')
            saving = True
        elif 'eferences' in text.lower() or 'ibliography' in text.lower():
            print(f'found reference, fontsize {fontsize}, y - {cm[5]}, {tm[5]}')
            saving = False

# %%
SAVING_FOLDER = 'texts'
if not os.path.exists(SAVING_FOLDER):
    os.makedirs(SAVING_FOLDER)
all_good_memories = []
# for file in files[:3]:
for file in files:
# for file in ['Kholodniuk_Bachelor_Thesis2021.pdf']:
    time_start = time.time()
    # read pdf
    pdfFileObj = open(os.path.join(PDFS_FOLDER, file), 'rb')
    pdf_reader = PyPDF2.PdfReader(pdfFileObj)
    print(f"{file} - {len(pdf_reader.pages)}")
    # save text
    saving = False
    section_names = []
    section_text = ""
    # Loop through the pages of the PDF file
    for page_num in range(len(pdf_reader.pages)):
        # Extract the text from the page
        page = pdf_reader.pages[page_num]
        text = page.extract_text(0, visitor_text=visitor)
        if saving:
            text = text[text.find(' '):].strip()
            section_text = section_text + text + ' '

    # clean up outliers
    temp_list = []
    for section in section_text.split('\n'):
        if section != '' and len(section.split()) > 10:
            temp_list.append(section)
    
    saver = " ".join(temp_list)
    # remove all occurances of \n
    saver = saver.replace('-\n', ' ')

    this_instance = [file[:-4], 
                     len(pdf_reader.pages),
                     len(temp_list),
                     len(saver),
                     len(saver.split())]
    all_good_memories.append(this_instance)
    # save as txt
    with open(os.path.join(SAVING_FOLDER, file[:-4]+'.txt'), 'w') as f:
        f.write(saver)
    print(f'Time taken: {round(time.time()-time_start, 2)}')
# %%
pd.DataFrame(all_good_memories, columns=['file', 'pages', 'sections', 'chars', 'words']).to_csv('df_raw.csv')
# %%
