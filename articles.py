from dataclasses import dataclass, field
from typing import List, Dict, Sequence


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