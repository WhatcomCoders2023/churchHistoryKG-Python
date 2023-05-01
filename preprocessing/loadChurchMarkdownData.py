import os
import glob


class LoadChurchData:

    def __init__(self, path_to_church_articles: str, entity_folder: str):
        self.path_to_church_articles = path_to_church_articles
        self.entity_folder = entity_folder
        self.relations_metadata_filename = f'{self.path_to_church_articles}/church-history-Article-en/Relations.csv'
        self.roles_metadata_filename = f'{self.path_to_church_articles}/church-history-Article-en/Roles.csv'
        self.relation_data_files = self._load_relation_data()
        self.markdown_data_files = self._load_markdown_data()
        self.entity_data = self._load_entity_data()

    def _load_relation_data(self):
        annotated_data = glob.glob(
            f'{self.path_to_church_articles}/church-history-Article-en/*.csv')
        annotated_data.remove(
            f'{self.path_to_church_articles}/church-history-Article-en/Relations.csv'
        )
        annotated_data.remove(
            f'{self.path_to_church_articles}/church-history-Article-en/Roles.csv'
        )
        return annotated_data

    def _load_entity_data(self):
        entity_data = glob.glob(
            f'{self.path_to_church_articles}/{self.entity_folder}/*.csv')
        return entity_data

    def _load_markdown_data(self):
        markdown_data = glob.glob(
            f'{self.path_to_church_articles}/church-history-Article-en/*.md')
        markdown_data.remove(
            f'{self.path_to_church_articles}/church-history-Article-en/ReadMe.md'
        )
        return markdown_data
