import json
from nltk.tokenize import sent_tokenize
'''
Function with opens json file to data without any entity and relation labels
1. For each sentence in json file
2. Create append an empty list to our entity and relation lists
'''


def format_dataset_to_predict(json_file_path: str):
    with open(json_file_path, 'r') as json_file:
        for line in json_file:
            json_data = json.loads(line.strip())
            json_data['ner'] = []
            json_data['relations'] = []
            for _ in json_data['sentences']:
                json_data['ner'].append([])
                json_data['relations'].append([])
    return json_data


def write_new_output(json_data: dict, output_file_path: str):
    with open(output_file_path, 'w') as output_file:
        json.dump(json_data, output_file)


'''
Function that takes in a long input of string, tokenizes string by sentence and formats into our dataset
'''


def convert_unstructured_text_to_dataset_input(text: str) -> dict:
    sentences = sent_tokenize(text)
    json_data = {}
    json_data['sentences'] = sentences
    json_data['ner'] = []
    json_data['relations'] = []
    for sent_idx, sentence in enumerate(json_data['sentences']):
        json_data['ner'].append([])
        json_data['relations'].append([])
    return json_data