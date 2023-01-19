from dataclasses import dataclass
from typing import List, Dict, Sequence


@dataclass
class ArticleSubsection:
    '''Dataclass class for article sub section (h3 tag)'''
    section_title: str
    bullet_points: List[str]


@dataclass
class ArticleSection:
    '''Dataclass class for article main section (h2 tag)'''
    section_title: str
    content: str
    bullet_points: List[str]
    article_subsections: Sequence[ArticleSubsection]


@dataclass
class ArticleSummary:
    '''Dataclass class for article summary section (h1 tag)'''
    title: str
    time_period: Dict[str, str]
    description: str
    summary: str


@dataclass
class Articles:
    '''Dataclass for Church History Articles '''
    level: int
    status: str
    identifer: str
    parent: str
    eras: List[str]
    summary: ArticleSummary
    article_sections: Sequence[ArticleSection]
