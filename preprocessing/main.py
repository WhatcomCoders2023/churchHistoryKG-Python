# Main class to run classes and prototype

from loadChurchMarkdownData import LoadChurchData
from markdown_preprocessing.processMarkdownData import MarkdownDataProcessor
from markdown_preprocessing.article_structs import ArticleReader
from dygiee_preprocessing.preprocess_faithlife_articles import PreprocessFaithlifeArticles
from faithlife_utils import load_faithlife_database_to_single_df


def main():
    faithlifeData = LoadChurchData(path_to_church_articles='faithlife_data',
                                   entity_folder='entities')
    processed_articles = MarkdownDataProcessor(faithlifeData,
                                               True).process_html()

    path_to_faithlife_db = 'faithlife_data/entities'
    database_df = load_faithlife_database_to_single_df(path_to_faithlife_db)
    articleReader = ArticleReader(processed_articles, database_df)
    # articleReader.merge_all_article_info(
    #     articleReader.name_to_articles['OrganizationUnderBishops'])
    # articleReader.write_all_articles_to_sentence_csv()
    output_path = "../faithlife_model_data/faithlife_data"

    preprocesser = PreprocessFaithlifeArticles(database_df, articleReader,
                                               output_path)
    preprocesser.process_articles()


main()
