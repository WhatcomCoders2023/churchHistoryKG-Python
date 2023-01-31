import csv

from dataclasses import dataclass, field
from typing import List, Sequence, Mapping
from nltk import tokenize


@dataclass
class ArticleSubsection:
    '''Dataclass class for article sub section (h3 tag)'''
    section_title: str = ""
    bullet_points: List[str] = field(default_factory=list)


@dataclass
class ArticleSection:
    '''Dataclass class for article main section (h2 tag)'''
    section_title: str = ""
    bullet_points: List[str] = field(default_factory=list)
    article_subsections: Sequence[ArticleSubsection] = field(
        default_factory=list)


@dataclass
class ArticleSummary:
    '''Dataclass class for article summary section (h1 tag)'''
    period: str = ""
    description: str = ""
    text: List[str] = field(default_factory=list)


@dataclass
class Articles:
    '''Dataclass for Church History Articles '''
    level: int = 0
    status: str = ""
    identifier: str = ""
    parent: str = ""
    eras: List[str] = field(default_factory=list)
    summary: ArticleSummary = ArticleSummary()
    article_sections: Sequence[ArticleSection] = field(default_factory=list)


class ArticleReader:
    """Class that manages reading and processing of all Bible Articles.

    Attributes:
        name_to_articles: Map of name of article to article object.
    """

    def __init__(self, articles: Articles) -> None:
        self.name_to_articles = self._map_articles_by_name(articles)

    def _map_articles_by_name(
            self, articles: Sequence[Articles]) -> Mapping[str, Articles]:
        """Creates a map of article names to articles
        
        Args:
            articles: List of all Bible Articles.

        Returns:
            A map of article name to article object.
        """
        name_to_articles = {}
        for article in articles:
            article_name = article.identifier
            name_to_articles[article_name] = article
        return name_to_articles

    def collect_all_sentences_in_article_by_section(
            self, article: Articles) -> Mapping[str, Sequence[str]]:
        """Extract all sentences from all section of an article

        Args:
            article: An article object.

        Returns:
            Map of section title to all sentences in each section.
        """
        sentences = {}

        summary_sentences = tokenize.sent_tokenize(article.summary.text)
        sentences['summary'] = summary_sentences
        for section in article.article_sections:
            section_name = section.section_title
            sentences[section_name] = []
            for section_bullet_point in section.bullet_points:
                section_bullet_point_sentences = tokenize.sent_tokenize(
                    section_bullet_point)
                sentences[section_name].append(section_bullet_point_sentences)

            for sub_section in section.article_subsections:
                sub_section_name = sub_section.section_title
                sentences[sub_section_name] = []
                for sub_section_bullet_point in sub_section.bullet_points:
                    sub_section_bullet_point_sentences = tokenize.sent_tokenize(
                        str(sub_section_bullet_point))
                    sentences[section_name].append(
                        sub_section_bullet_point_sentences)
        return sentences
