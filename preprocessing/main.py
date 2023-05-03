# Main class to run classes and prototype

from loadChurchMarkdownData import LoadChurchData
from markdown_preprocessing.processMarkdownData import MarkdownDataProcessor
from markdown_preprocessing.article_structs import ArticleReader
from dygiee_preprocessing.PreprocessFaithlifeArticles import PreprocessFaithlifeArticles


def main():
    faithlifeData = LoadChurchData(path_to_church_articles='faithlife_data',
                                   entity_folder='entities')
    processed_articles = MarkdownDataProcessor(faithlifeData).process_html()
    articleReader = ArticleReader(processed_articles)
    path_to_faithlife_db = 'faithlife_data/entities'
    output_path = "../faithlife_model_data/faithlife_data"

    preprocesser = PreprocessFaithlifeArticles(path_to_faithlife_db,
                                               articleReader, output_path)
    preprocesser.process_articles()


main()
