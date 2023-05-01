import json
import pandas as pd

from typing import List, Tuple
from article_structs import ArticleReader, Article, NERTuple
from faithlife_utils import load_faithlife_database_to_single_df, make_faithlife_map, search_entity_type_in_faithlife_map_with_entity_label, search_entity_in_faithlife_map, label_events
from preprocessing_utils import replace_special_chars_in_entity_annotations, tokenize_sentence, tokenize_entity
from pathlib import Path


class PreprocessFaithlifeArticles:

    def __init__(self,
                 path_to_faithlife_db: str,
                 articleReader: ArticleReader,
                 output_path: str = 'faithlife_data'):
        self.articleReader = articleReader
        self.database_df = load_faithlife_database_to_single_df(
            path_to_faithlife_db)

        self.entity_label_to_entity_type = make_faithlife_map(self.database_df)
        self.output_path = output_path

    def process_articles(self):
        all_docs = []
        for article_name, article in self.articleReader.name_to_articles.items(
        ):
            # Temp until chicago manual of style is fixed
            self.articleReader.filter_article_section(article,
                                                      'Recommended Reading')
            all_sentences, ner_tuples = self.articleReader.merge_all_article_info(
                article)

            ner_tuples = self.label_ner_for_sentences(ner_tuples, all_sentences,
                                                      article_name)
            tokenized_sentences = self.tokenize_sentences(all_sentences)
            ner_tuples = self.convert_character_indices_to_sentence_indices(
                tokenized_sentences, ner_tuples)
            ner_tuples = label_events(ner_tuples, article.metadata.identifier)
            ner_tuples = self.remove_labeled_sentence_tuple(ner_tuples)
            print("after removed tuples:", ner_tuples)

            relation_map = self.search_csv_for_relation(article_name,
                                                        tokenized_sentences,
                                                        ner_tuples)

            article = self.populate_relation_field_for_article(
                article, relation_map)
            article.ner_tuples = ner_tuples
            article.tokenized_sentences = tokenized_sentences

            self.articleReader.save_article_to_csv(article)
            self.articleReader.transform_csv_to_json_data(article)

            # write to jsonl
            output_json_dict = {}
            output_json_dict["doc_key"] = article.metadata.identifier
            output_json_dict["sentences"] = article.tokenized_sentences
            ner_tuples = article.ner_tuples
            unpacked_ner_tuples = [
                [tuple(nt) for nt in inner_list] for inner_list in ner_tuples
            ]

            output_json_dict["ner"] = unpacked_ner_tuples
            output_json_dict["relations"] = article.relations

            all_docs.append(output_json_dict)

            self.write_labeled_data_to_json(all_docs)

    def get_sentence_indices(
        self,
        prev_sent_index: int,
        tokenized_sentence: List[str],
        ner_tuple: NERTuple,
    ) -> NERTuple:
        ''' Create new tuple with sentence indices swapped out for character indices. 

        TODO - Investiage robustness, sometimes tokenization is off and this relies on 1:1 match
        for example, if it needs to find '1475', but word is tokenized as 'd. 1475'. Right now, we
        can just throw away labels

        Args:
            tokenized_sentence: List of tokens corresponding to a sentence from the article.
            labeled_sentence: Tuple containing ner information in the following format

            Format - start_index, end_index, entity_type, entity_label, sentence
            Ex - [0, 24, 'NONE', 'The Renaissance in Europe', 'bk.tle%3arenaissanceeurope']

        Returns:
            Tuple where char indices are replaced with sentence indices


        Ex:
        tokenized_sentence - ['Schism', 'separated', 'the', 'Russian', 'Orthodox', 'Church', 'and', 
        'the', 'Old', 'Believers', 'over', 'church', 'reforms', '(', '1652–1725', ')', '.']

        labeled_sentence: [53, 65, 'concept', 'bk.%Raskolniki', 'Old Believers']
        sentence_indices: [8, 9, 'concept', 'bk.%Raskolniki', 'Old Believers']

        **lowercasing is applied to line up with database
        '''
        print("tokenized_sentnece:", tokenized_sentence)
        lower_case_tokenized_sentence = [
            word.lower() for word in tokenized_sentence
        ]
        print("tokenized_sentence:", lower_case_tokenized_sentence)
        print('ner_tuple:', ner_tuple)
        entity = ner_tuple.entity_name
        print("entity:", entity)

        # print('entity:', entity)
        tokenized_entity = tokenize_entity(entity)
        lower_case_tokenized_entity = [
            word.lower() for word in tokenized_entity
        ]
        # entity = entity.split(" ")
        print('new_entity:', lower_case_tokenized_entity)
        entity_index = 0
        total_entity_words = len(lower_case_tokenized_entity)
        print('total_entity_words', total_entity_words)
        start_entity_sent = 0
        end_entity_sent = 0

        if is_subsequence(lower_case_tokenized_entity,
                          lower_case_tokenized_sentence):
            start_entity_sent, end_entity_sent = 0, len(
                lower_case_tokenized_entity) - 1
        else:
            return NERTuple(start_entity_sent + prev_sent_index,
                            end_entity_sent + prev_sent_index,
                            ner_tuple.entity_type, ner_tuple.entity_label,
                            ner_tuple.entity_name)

            # return ([
            #     start_entity_sent + prev_sent_index,
            #     end_entity_sent + prev_sent_index, ner_tuple.entity_type,
            #     ner_tuple.entity_label, ner_tuple.entity_name
            # ])

        for i in range(
                len(lower_case_tokenized_sentence) -
                len(lower_case_tokenized_entity) + 1):
            print(
                'searching for:',
                lower_case_tokenized_sentence[i:i +
                                              len(lower_case_tokenized_entity)])

            if lower_case_tokenized_sentence[
                    i:i + len(lower_case_tokenized_entity
                             )] == lower_case_tokenized_entity:
                start_entity_sent, end_entity_sent = i, i + len(
                    lower_case_tokenized_entity) - 1
                print('start:', start_entity_sent + prev_sent_index, 'end:',
                      end_entity_sent + prev_sent_index)

                return NERTuple(start_entity_sent + prev_sent_index,
                                end_entity_sent + prev_sent_index,
                                ner_tuple.entity_type, ner_tuple.entity_label,
                                ner_tuple.entity_name)
                # return ([
                #     start_entity_sent + prev_sent_index,
                #     end_entity_sent + prev_sent_index, ner_tuple.entity_type,
                #     ner_tuple.entity_label, ner_tuple.entity_name
                # ])

        # tokenized_entity = tokenize_entity(labeled_sentence[4])
        # print("tokenized_entity:", tokenized_entity)
        # for i in range(len(tokenized_sentence) - len(tokenized_entity) + 1):
        #     if tokenized_sentence[i:i +
        #                           len(tokenized_entity)] == tokenized_entity:
        #         start_entity_sent, end_entity_sent = i, i + len(
        #             tokenized_entity) - 1
        #         return ([
        #             start_entity_sent + prev_sent_index,
        #             end_entity_sent + prev_sent_index, labeled_sentence[2],
        #             labeled_sentence[3], labeled_sentence[4]
        #         ])

        print(f'did not find: {tokenized_entity} in {entity}')
        return None

    def search_for_corresponding_obj_label(
            self, obj_relation: str, ner_tuples: List[NERTuple]) -> NERTuple:
        '''Search for a corresponding obj label in ner_tuples to complete relation extraction.

        This presumes that a subj label exists in our ner_tuples so we attempt to complete the relation
        by finding a corresponding obj_label in ner_tuples.

        Args:
            obj_relation: Entity label for obj relation from csv file.
            ner_tuples: List of tuples where indices are [start, end, entity_type, entity_label, sentence]

        Returns:
            If it exists, the ner_tuple which has a corresponding obj_label that matches subj_label
            to complete the relation.
        
        
        '''
        for ner_tuple in ner_tuples:
            if ner_tuple.entity_label == obj_relation:
                return ner_tuple
        return ()

    def remove_labeled_sentence_tuple(
        self,
        ner_tuples: List[List[NERTuple]],
    ) -> List[List[NERTuple]]:
        '''Removes labeled ner's (labeled_sentence tuple) with following conditions
        
        Conditions to remove:
        1) Event, denom, denomgroup entity types 
        2) The sentence indices are -1, which means we could not properly
        map character indices to sentence indices
        3) The entity span is longer than 8 words

        article.labeled_sentence = [sent_start_index, sent_end_index, entity_type, entity_label, sentence]
        
        Args:
            article: Data structure for faithlife article data.

        Returns:
            article data structure where labeled sentence indices removed that were
            filtered out.
        '''

        for i, ner_tuples_in_sentence in enumerate(ner_tuples):
            index_to_delete = []
            for index, ner_tuple in enumerate(ner_tuples_in_sentence):

                # condition 1
                if ner_tuple.entity_type in [
                        'NONE', 'event', 'denom', 'denomgroup', 'thing'
                ]:
                    index_to_delete.append(index)

                # condition 2
                if ner_tuple.start_index == -1 or ner_tuple.end_index == -1:
                    index_to_delete.append(index)

                # condition 3
                if ner_tuple.end_index - ner_tuple.start_index > 8:
                    index_to_delete.append(index)

            new_ner_tuples = [
                i for j, i in enumerate(ner_tuples_in_sentence)
                if j not in index_to_delete
            ]
            ner_tuples[i] = new_ner_tuples
        return ner_tuples

    def label_ner_for_sentences(self, ner_tuples: List[List[NERTuple]],
                                sentences: List[str],
                                article_name: str) -> List[List[NERTuple]]:
        '''Labels NER for sentences in Article using 3 methods.

        1) href hyperlinks in the html document - These are already in article.ner_tuples
        this method searches the faithlife database to find the corresponding entity type.

        2) Search for occurrence of NER in sentence using labeled csv file for article (unused)

        3) Iterate through every phrase in the faithlife database and search the given sentence
        to find a match (computationally expensive).


        Args:
            article: Data structure for faithlife article data.
            article_name: Identifier of the article.

        Returns:
            Article data structure with labeled sentence where

        Example Output:
        sentence: "The Protestant Reformation challenged traditional sources of authority
        and religious dogma (1517–1648)."

        labeled_sentence:  [[62, 70, "concept", "bk.%authority", "authority"], 
                            [77, 12, "concept", "bk.%dogma", "dogma"]]

        '''
        no_relations = [
            'OrganizationUnderBishops', 'RiseNaziGermanyDividesChurch',
            'WorshipEarlyChurch', 'Pelagian', "ChristianApologists",
            "ChurchAndState", 'SporadicPersecution', 'ChineseChurchGrows',
            'Puritanism'
        ]
        ner_tuples = self.find_entity_type_for_href_ner_tuples_in_article_struct(
            ner_tuples)

        if article_name not in no_relations:
            relations_df = pd.read_csv(
                f'faithlife_data/church-history-Articles-en/{article_name}.csv')
            subj_list = set(relations_df['subj'].tolist())
            obj_list = set(relations_df['obj'].tolist())
            all_entities_in_history_article_csv = subj_list.union(obj_list)

            ner_tuples = self.find_and_add_entities_from_history_article_labels_to_article_struct(
                ner_tuples, sentences, all_entities_in_history_article_csv)

        return ner_tuples

    def find_entity_type_for_href_ner_tuples_in_article_struct(
            self, ner_tuples: List[List[Tuple]]) -> List[List[NERTuple]]:
        ner_tuples_with_entity_type = []
        for i, ner_labels_from_href_in_html in enumerate(ner_tuples):
            print("ner_labels_from_href_in_html", ner_labels_from_href_in_html)

            ner_tuples_in_sentence = []
            for sent_idx, ner_tuple_from_html in enumerate(
                    ner_labels_from_href_in_html):
                print("ner_tuple_from_html:", ner_tuple_from_html)
                entity_label = replace_special_chars_in_entity_annotations(
                    ner_tuple_from_html[3])

                # second test if first did not find anything
                entity_type = search_entity_type_in_faithlife_map_with_entity_label(
                    entity_label, self.entity_label_to_entity_type)

                # ner_label = [
                #     ner_label[0], ner_label[1], entity_type, ner_label[3],
                #     ner_label[2]
                # ]  #remove later
                created_ner_tuple = NERTuple(ner_tuple_from_html[0],
                                             ner_tuple_from_html[1],
                                             entity_type, entity_label,
                                             ner_tuple_from_html[2])
                ner_tuples_in_sentence.append(created_ner_tuple)
                '''
                1) start_index
                2) end_index
                3) entity_type
                4) entity label
                5) sentence
                '''
            ner_tuples_with_entity_type.append(ner_tuples_in_sentence)

        return ner_tuples_with_entity_type

    def find_and_add_entities_from_history_article_labels_to_article_struct(
            self, ner_tuples: List[List[NERTuple]], sentences: List[str],
            all_entities_in_history_article_csv: set[str]) -> List[List[Tuple]]:

        for i in range(len(sentences)):
            # search database for more labels
            # print(
            #     f'\n\nsearching entities for sentence: {article.sentences[i]}')
            # print(f'Existing entities: {article.ner_tuples}')
            db_entities = search_entity_in_faithlife_map(
                sentences[i], ner_tuples[i], self.database_df,
                all_entities_in_history_article_csv)

            ner_tuples[i].extend(db_entities)

            if ner_tuples[i]:
                print("curr_tuple:", ner_tuples[i])

                #previous statement appends new elements to end of list, must sort
                ner_tuples[i].sort(key=lambda ner_tuple:
                                   (ner_tuple.start_index, ner_tuple.end_index))
        return ner_tuples

    def tokenize_sentences(self, sentences: List[str]) -> List[List[str]]:
        '''Tokenizes sentences in article.

        Args:
            article: Data structure for faithlife article data.

        Returns:
        '''
        tokenized_sentences = []
        for i, sentence in enumerate(sentences):

            tokenized_sentence = tokenize_sentence(sentence)  #
            print(f'sentence{i}: {tokenized_sentence}')
            tokenized_sentences.append(tokenized_sentence)
        return tokenized_sentences

    def convert_character_indices_to_sentence_indices(
        self,
        tokenized_sentences: List[List[str]],
        ner_tuples: List[List[NERTuple]],
    ) -> List[List[NERTuple]]:
        '''Converts character indices of entity to sentence indices.

        Args:
            article: Data structure for faithlife article data.

        Returns:
            Article data structure with labeled sentence tuple overidden with sentence indices


        Example Output:
        sentence: "The Protestant Reformation challenged traditional sources of authority
        and religious dogma (1517–1648)."

        previous_labeled_sentence: [[62, 70, "concept", "bk.%authority", "authority"], 
                                    [77, 12, "concept", "bk.%dogma", "dogma"]]
        
        labeled_sentence:  [[7, 7, "concept", "bk.%authority", "authority"], 
                            [10, 10, "concept", "bk.%dogma", "dogma"]]
        '''

        # labeled sentences is a bad named, because it implies multiple sentences
        # but its actually just 1 sentence with multiple ner labels

        ner_tuples_processed = []
        last_sent_index = 0
        for i, ner_tuples in enumerate(ner_tuples):
            ner_tuple_in_sentence = []

            for j, ner_tuple in enumerate(ner_tuples):

                ner_tuples_with_sentence_indices = self.get_sentence_indices(
                    last_sent_index, tokenized_sentences[i], ner_tuple)
                print('sent_indices:', ner_tuples_with_sentence_indices)

                if ner_tuples_with_sentence_indices != None:
                    ner_tuple_in_sentence.append(
                        ner_tuples_with_sentence_indices)
                    #fix later, should map to a new section and not override previous one
            last_sent_index += len(tokenized_sentences[i])
            ner_tuples_processed.append(ner_tuple_in_sentence)

        return ner_tuples_processed

    def populate_relation_field_for_article(
            self, article: Article, relation_map: dict[str, str]) -> Article:
        for sentence_index, relations in relation_map.items():
            article.relations.append([])
            for relation in relations:
                article.relations[sentence_index].append(relation)
        return article

    def search_csv_for_relation(
        self,
        article_name: str,
        tokenized_sentences: List[str],
        ner_tuples: List[List[NERTuple]],
    ) -> dict[int, List[str]]:
        '''Use article's annotation to extract relations.
        
        Given an article's annotation, we iterate through the ner_tuples and find rows which
        align with the article's annotation. Then, we attempt to find a relation between 
        subject and object. So, we assert that for a subject that there exists an entity label
        in our ner_tuples with a corresponding object type that matches a given relation
        

        ner_tuple = [start_index, end_index, entity_type, entity_label, sentence]
        Args:
            article: Data structure for faithlife article data.
            article_name: Identifier of article.

        Returns:
            Article with relations field filled out.
        '''

        no_relations = [
            'OrganizationUnderBishops', 'RiseNaziGermanyDividesChurch',
            'WorshipEarlyChurch', 'Pelagian', "ChristianApologists",
            "ChurchAndState", 'SporadicPersecution', 'ChineseChurchGrows',
            'Puritanism'
        ]
        binary_flag = True  #training
        relation_map = {}
        if article_name not in no_relations:
            relations_df = pd.read_csv(
                f'faithlife_data/church-history-Articles-en/{article_name}.csv')

            relation_map = {i: [] for i in range(len(tokenized_sentences))
                           }  #cache sentence length
            for sentence_index, ner_tuples_in_sentence in enumerate(ner_tuples):
                for ner_tuple in ner_tuples_in_sentence:
                    cells_where_entity_label_exists_as_subj = relations_df.loc[
                        relations_df['subj'] == ner_tuple.entity_label]
                    for _, t in cells_where_entity_label_exists_as_subj.iterrows(
                    ):
                        obj_tuple = self.search_for_corresponding_obj_label(
                            t.obj, ner_tuples_in_sentence)

                        if obj_tuple and binary_flag:
                            relation_map[sentence_index].append([
                                ner_tuple.start_index, ner_tuple.end_index,
                                obj_tuple.start_index, obj_tuple.end_index,
                                "rel:TRUE"
                            ])
                        elif obj_tuple:
                            relation_map[sentence_index].append([
                                ner_tuple.start_index, ner_tuple.end_index,
                                obj_tuple.start_index, obj_tuple.end_index,
                                t.rel
                            ])
                        else:
                            pass

        return relation_map

    def write_labeled_data_to_json(self, all_docs: list) -> None:
        Path('../faithlife_model_data').mkdir(parents=True, exist_ok=True)
        with open(f'{self.output_path}.jsonl', 'w') as outfile:
            for entry in all_docs:
                json.dump(entry, outfile, ensure_ascii=False)
                outfile.write('\n')


def is_subsequence(subseq, sentence):
    subseq_pos = 0  # Position of next character to match in subsequence
    for char in sentence:
        if char == subseq[subseq_pos]:
            subseq_pos += 1
            if subseq_pos == len(subseq):  # Reached end of subsequence
                return True
    return False