# ==================================================
# File: setup_data.py
# Author: Trent Bultsma
# Date: 12/12/2022
# Description: Unzips the dataset then removes 
#   hashtags, tagging people, and urls from it.
# ==================================================

import zipfile
import re

# unzip the dataset
with zipfile.ZipFile('dataset.zip', 'r') as zip:
    zip.extractall('./')

# sanitize the datset
with open('dataset.csv', 'r', encoding='utf-8') as dataset:
    data = ''.join(dataset.readlines())
    data = re.sub(r'#\w+', '', data) # remove hashtags
    data = re.sub(r'@\w+', '', data) # remove tagging people
    data = re.sub(r'(http)[\w/\.:]+', '', data) # remove urls

    # write the sanitized data
    with open('dataset_sanitized.csv', 'w', encoding='utf-8') as dataset_sanitized:
        dataset_sanitized.write(data)