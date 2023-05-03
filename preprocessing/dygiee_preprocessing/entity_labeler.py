import pandas as pd
from typing import List, Tuple, Mapping

from markdown_preprocessing.article_structs import NERTuple
from faithlife_utils import search_entity_in_faithlife_map, search_entity_type_in_faithlife_map_with_entity_label
from dygiee_preprocessing.faithlife_tokenizer import FaithlifeTokenizer
from preprocessing_utils import replace_special_chars_in_entity_annotations


class EntityLabeler:

    def __init__(
        self,
        tokenizer: FaithlifeTokenizer,
        entity_label_to_entity_type: Mapping[str, str],
        database_df: pd.DataFrame,
    ):
        self.tokenizer = tokenizer
        self.entity_label_to_entity_type = entity_label_to_entity_type
        self.database_df = database_df
        self.no_relations = [
            'OrganizationUnderBishops', 'RiseNaziGermanyDividesChurch',
            'WorshipEarlyChurch', 'Pelagian', "ChristianApologists",
            "ChurchAndState", 'SporadicPersecution', 'ChineseChurchGrows',
            'Puritanism'
        ]

    def label_ner_for_sentences(
        self,
        ner_tuples: List[List[NERTuple]],
        sentences: List[List[str]],
        article_name: str,
    ) -> List[List[NERTuple]]:
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

        ner_tuples = self.find_entity_type_for_href_ner_tuples_in_article_struct(
            ner_tuples)

        ner_tuples = self.use_history_article_labels(ner_tuples, sentences,
                                                     article_name)
        return ner_tuples

    def use_history_article_labels(
        self,
        ner_tuples: List[List[NERTuple]],
        sentences: List[str],
        article_name: str,
    ):
        if article_name not in self.no_relations:
            relations_df = pd.read_csv(
                f'faithlife_data/church-history-Articles-en/{article_name}.csv')
            subj_list = set(relations_df['subj'].tolist())
            obj_list = set(relations_df['obj'].tolist())
            all_entities_in_history_article_csv = subj_list.union(obj_list)

            ner_tuples = self.find_and_add_entities_from_history_article_labels_to_article_struct(
                ner_tuples, sentences, all_entities_in_history_article_csv)
        return ner_tuples

    def find_entity_type_for_href_ner_tuples_in_article_struct(
        self,
        ner_tuples: List[List[Tuple]],
    ) -> List[List[NERTuple]]:
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
        self,
        ner_tuples: List[List[NERTuple]],
        sentences: List[str],
        all_entities_in_history_article_csv: set[str],
    ) -> List[List[Tuple]]:

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
        tokenized_entity = self.tokenizer.tokenize_entity(entity)
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

        if self.is_subsequence(lower_case_tokenized_entity,
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

    def label_events(
        self,
        ner_tuples: List[List[NERTuple]],
        article_identifer: str,
    ) -> List[List[NERTuple]]:
        print("label_events", ner_tuples)
        for i, ner_tuple_in_sentence in enumerate(ner_tuples):
            for j, ner_tuple in enumerate(ner_tuple_in_sentence):
                if i == 0 and j == 0 and ner_tuple.entity_name == article_identifer:  # Overwrites event, bc in this context its refering to a document
                    ner_tuples[i] = []  # only one tuple for the title
                    ner_tuple.entity_type = 'writing'
                    ner_tuples[i].append(ner_tuple)
                    print("check this tuple:", ner_tuples[i][j])
                elif ner_tuple.entity_type == 'NONE':
                    if 'bk.tle%' in ner_tuple.entity_label:  #this is wrong -- should be event not book
                        ner_tuple.entity_type = 'event'
        return ner_tuples

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
                        'NONE', 'event', 'denom', 'denomgroup', 'thing',
                        'artifact'
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

    def is_subsequence(
        self,
        subseq,
        sentence,
    ):
        subseq_pos = 0  # Position of next character to match in subsequence
        for char in sentence:
            if char == subseq[subseq_pos]:
                subseq_pos += 1
                if subseq_pos == len(subseq):  # Reached end of subsequence
                    return True
        return False