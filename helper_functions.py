import pandas as pd
import json
import ast

def parse_and_clean(text):
    if pd.isna(text) or text == '[]': 
        return ''
    try:
        
        list_of_dicts = json.loads(text)
        return ' '.join([d['name'] for d in list_of_dicts if 'name' in d])
    except (json.JSONDecodeError, TypeError):
        return str(text).replace('[', '').replace(']', '').replace("'", "").replace('"', '').replace(',', ' ').strip()


