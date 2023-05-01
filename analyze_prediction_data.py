import json
import pathlib
import random
from typing import List

prediction_path = 'prediction_data/'

# [[0, 0, 'concept', 'bk.%pagans', 'Paganism']]


def read_article(prediction_folder: str):

    with open(prediction_folder, 'r') as f:
        for line in f:
            json_obj = json.loads(line.strip())
            doc_key = json_obj['doc_key']
            print("article name:", doc_key)
            sentence_index = 0

            for sent_idx, sentence in enumerate(json_obj['sentences']):
                for predicted_entity_tuples in json_obj['predicted_ner'][
                        sent_idx]:
                    predicted_entity_tuples.append(
                        sentence[predicted_entity_tuples[0] -
                                 sentence_index:predicted_entity_tuples[1] -
                                 sentence_index + 1])
                    print("predicted entities:", predicted_entity_tuples)
                for entity_tuples in json_obj['ner'][sent_idx]:
                    print("real entities:", entity_tuples)
                sentence_index += len(sentence)
                print("sentence_index:", sentence_index)


read_article(prediction_path + 'run_7.json')
