import os
import glob


class LoadChurchData:

    def __init__(self, path_to_church_articles: str):
        if os.getcwd() != path_to_church_articles:
            os.chdir(path_to_church_articles)

        self.path_to_church_articles = path_to_church_articles
        self.relations_metadata_filename = 'Relations.csv'
        self.roles_metadata_filename = 'Roles.csv'
        self.relation_data_files = self._load_relation_data()
        self.markdown_data_files = self._load_markdown_data()

    def _load_relation_data(self):
        annotated_data = glob.glob('*{}'.format('.csv'))
        annotated_data.remove('Relations.csv')
        annotated_data.remove('Roles.csv')
        return annotated_data

    def _load_markdown_data(self):
        markdown_data = glob.glob("*{}".format('.md'))
        markdown_data.remove('ReadMe.md')
        return markdown_data
