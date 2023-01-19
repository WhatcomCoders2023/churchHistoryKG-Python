import markdown
import bs4

from loadChurchMarkdownData import LoadChurchData
from articles import *
from typing import Sequence


class MarkdownDataProcessor:
    '''MarkdownDataProcessors converts markdown data to dataclasses.

    Attributes:
        dataloader: Class that loads Markdown data.
        html_data: List of html strings, from converted markdown files.
    '''

    def __init__(self, dataloader: LoadChurchData):
        self.dataloader = dataloader
        self.html_data = self.markdown_to_html()

    def markdown_to_html(self) -> Sequence[str]:
        '''Converts markdown files to html strings.

        Returns:
            List of html strings, from converted markdown files
        '''
        html_data = []
        for markdown_file in self.dataloader.markdown_data_files:
            f = open(markdown_file, 'r')
            html_from_markdown = markdown.markdown(f.read())
            html_data.append(html_from_markdown)
        return html_data

    def process_html(self) -> Sequence[Articles]:
        '''Processes list of html strings and converts to dataclasses.
        
        Returns:
            List of Articles Dataclasses
        '''
        articles = []
        for html in self.html_data:
            soup = bs4.BeautifulSoup(html, "html.parser")
            article = Articles()

            metadata = self.parse_metadata_from_article(soup)
            article = self.get_data_from_metadata(article, metadata)

            h1_header = self.parse_summary_from_article(soup)
            article = self.get_data_from_h1_header(article, h1_header)

            h2_header = self.parse_article_sections(soup)
            article = self.get_data_from_h2_header(article, h2_header)
            articles.append(article)
        return articles

    def parse_metadata_from_article(self,
                                    soup: bs4.BeautifulSoup) -> bs4.element.Tag:
        '''Parses metadata information from markdown article.

        Args:
            soup: Beautiful Soup Module used to parse html data.
        '''
        metadata = soup.find_all('p')[0]
        return metadata

    def parse_summary_from_article(self,
                                   soup: bs4.BeautifulSoup) -> bs4.element.Tag:
        '''Parses summary information from markdown article.
        
        Args:
            soup: Beautiful Soup Module used to parse html data.
        '''
        h1_header = soup.find_all('h1')[0]
        return h1_header

    def parse_article_sections(self,
                               soup: bs4.BeautifulSoup) -> bs4.element.Tag:
        '''Parses article section information from markdown article.
        
        Args:
            soup: Beautiful Soup Module used to parse html data.
        '''
        h2_headers = soup.find_all('h2')
        return h2_headers

    def get_data_from_metadata(self, article: Articles,
                               metadata: bs4.element.Tag) -> Articles:
        '''Gets data from metadata tag and stores in Article Dataclass.

        Args:
            article: Articles dataclass.
            metadata: Bs4 tag containing <p> element corresponding to 
            the articles metadata.
        '''
        metadata_attributes = metadata.text.split("\n")
        for attribute in metadata_attributes:
            metadata_values = attribute.split(": ")
            metadata_key, metadata_value = metadata_values[0].lower(
            ), metadata_values[1:]

            if metadata_key == 'level':
                article.level = metadata_value[0]
            elif metadata_key == 'status':
                article.status = metadata_value[0]
            elif metadata_key == 'identifier':
                article.identifier = metadata_value[0]
            elif metadata_key == 'parents':
                article.parents = metadata_value[0]
            elif metadata_key == 'eras':
                article.eras = metadata_value[:]

        return article

    def get_data_from_h1_header(self, article: Articles,
                                h1_header: bs4.element.Tag) -> Articles:
        '''Gets data from summary tag and stores in Article Dataclass.
        
        Args:
            article: Articles dataclass.
            metadata: Bs4 tag containing <h1> element corresponding to 
            the article's summary section.
        '''
        article_header = h1_header.text

        article_html_list = h1_header.find_next()
        article_header, article_html_list

        all_lists = article_html_list.findAll('li')
        for in_list in all_lists:
            in_list_values = in_list.text.split(": ")
            in_list_key, in_list_value = in_list_values[0].lower(
            ), in_list_values[1]
            if in_list_key == 'period':
                #todo - regex replace !!smallcaps|ad!! -> AD
                article.summary.period = in_list_value
            elif in_list_key == 'description':
                article.summary.description = in_list_value
        return article

    def get_data_from_h2_header(self, article: Articles,
                                h2_headers_html: bs4.element.Tag) -> Articles:
        '''Gets data from section tag and stores in Article Dataclass.
        
        Args:
            article: Articles dataclass.
            metadata: Bs4 tag containing <h2> element corresponding to 
            the article's sections.
        '''
        for i, h2_header in enumerate(h2_headers_html):
            if i == 0:
                continue
            header_name = h2_header.text

            articleSection = ArticleSection()
            articleSection.section_title = header_name
            h2_article_section = h2_header.find_next('ul')

            all_lists = h2_article_section.findAll('li')
            for in_list in all_lists:
                in_list_values = in_list.text
                articleSection.bullet_points.append(in_list_values)

            h3_headers = h2_header.find_next_sibling('h3')

            while h3_headers:
                articleSubsection = ArticleSubsection()
                articleSubsection.section_title = h3_headers.text
                subsection = h3_headers.find_next('ul')

                all_lists = subsection.findAll('li')
                for in_list in all_lists:
                    in_list_values = in_list.text
                    articleSubsection.bullet_points.append(in_list)
                articleSection.article_subsections.append(articleSubsection)
                h3_headers = h3_headers.find_next('h3')
            article.article_sections.append(articleSection)
        return article
