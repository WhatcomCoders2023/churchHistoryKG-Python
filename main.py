# Main class to run classses and prototype

from loadChurchMarkdownData import LoadChurchData
from processMarkdownData import MarkdownDataProcessor
from articles import ArticleReader


def main():

    data = LoadChurchData('church-history-articles-en')
    articles = MarkdownDataProcessor(data).process_html()
    reader = ArticleReader(articles)

    article = reader.name_to_articles['SpreadsToEurope']
    for section in article.article_sections:
        for subsection in section.article_subsections:
            print(subsection.section_title)
            print(subsection.bullet_points)