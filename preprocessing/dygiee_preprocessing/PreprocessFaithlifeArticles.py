import pandas as pd

from markdown_preprocessing.article_structs import ArticleReader
from faithlife_utils import load_faithlife_database_to_single_df, make_faithlife_map
from dygiee_preprocessing.faithlife_tokenizer import FaithlifeTokenizer
from dygiee_preprocessing.entity_labeler import EntityLabeler
from dygiee_preprocessing.relation_labeler import RelationLabeler
from dygiee_preprocessing.dygiee_writer import DYGIEEWriter


class PreprocessFaithlifeArticles:

    def __init__(
        self,
        path_to_faithlife_db: str,
        articleReader: ArticleReader,
        output_path: str = 'faithlife_data',
    ):
        self.articleReader = articleReader
        self.database_df = load_faithlife_database_to_single_df(
            path_to_faithlife_db)

        self.entity_label_to_entity_type = make_faithlife_map(self.database_df)
        self.output_path = output_path
        self.tokenizer = FaithlifeTokenizer()
        self.entity_labeler = EntityLabeler(self.tokenizer,
                                            self.entity_label_to_entity_type,
                                            self.database_df)
        self.relation_labeler = RelationLabeler()
        self.dygiee_writer = DYGIEEWriter("../faithlife_model_data")

    def process_articles(self):
        for article_name, article in self.articleReader.name_to_articles.items(
        ):
            # Temp until chicago manual of style is fixed
            self.articleReader.filter_article_section(article,
                                                      'Recommended Reading')
            all_sentences, ner_tuples = self.articleReader.merge_all_article_info(
                article)

            tokenized_sentences = self.tokenizer.tokenize_article_sentences(
                all_sentences)

            ner_tuples = self.entity_labeler.label_ner_for_sentences(
                ner_tuples, tokenized_sentences, article_name)

            ner_tuples = self.entity_labeler.convert_character_indices_to_sentence_indices(
                tokenized_sentences, ner_tuples)

            ner_tuples = self.entity_labeler.label_events(
                ner_tuples, article.metadata.identifier)

            ner_tuples = self.entity_labeler.remove_labeled_sentence_tuple(
                ner_tuples)

            relation_map = self.relation_labeler.search_csv_for_relation(
                article_name, tokenized_sentences, ner_tuples)
            relation_tuples = self.relation_labeler.populate_relation_field_for_article(
                relation_map)
            print(relation_tuples)

            #TODO - Maybe delete later?
            article.ner_tuples = ner_tuples
            article.tokenized_sentences = tokenized_sentences
            article.relations = relation_tuples

            self.articleReader.save_article_to_csv(article)
            self.articleReader.transform_csv_to_json_data(article)
            output_docs = self.dygiee_writer.write_to_jsonl(article)

            self.dygiee_writer.write_labeled_data_to_json(
                output_docs, article.metadata.identifier)
