# Main class to run classes and prototype

import json
import random

from loadChurchMarkdownData import LoadChurchData
from processMarkdownData import MarkdownDataProcessor
from Article import ArticleReader
from PreprocessFaithlifeArticles import PreprocessFaithlifeArticles


def split_jsonl_file(input_path: str):

    print("test")
    with open(input_path, 'r') as f:
        for line in f:
            print(line)
            json_obj = json.loads(line.strip())
            # yield json_obj
            print(json_obj)


def main():
    faithlifeData = LoadChurchData(path_to_church_articles='faithlife_data',
                                   entity_folder='entities')
    Article = MarkdownDataProcessor(faithlifeData).process_html()
    articleReader = ArticleReader(Article)
    path_to_faithlife_db = 'faithlife_data/entities'
    output_path = "../faithlife_model_data/faithlife_data"

    preprocesser = PreprocessFaithlifeArticles(path_to_faithlife_db,
                                               articleReader, output_path)
    preprocesser.process_articles()


main()
# input_path = 'faithlife_data/processed_data/faithlife_data.jsonl'
# split_jsonl_file(input_path)
