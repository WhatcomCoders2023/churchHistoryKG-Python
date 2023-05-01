import pandas as pd
import re
from typing import List, Tuple, Set, Mapping
from preprocessing_utils import replace_special_chars_in_entity_annotations
from articles import NERTuple


def load_faithlife_database_to_single_df(
        path_to_faithlife_db_folder: str) -> pd.DataFrame:
    '''Loads faithlife concept database to a single panda dataframe.

    Args:
        path_to_faithlife_db_folder: Directory of faithlife database csv files.

    Returns:
        Pandas dataframe that contains faithlife concept database.
    '''

    file_names = [
        'agent', 'bio', 'era', 'event', 'lco', 'lcv', 'lsto', 'place', 'thing',
        'tle'
    ]
    column_names = ['kind', 'primary', 'label_en', 'rank']

    database_df = pd.concat([
        pd.read_csv(f'{path_to_faithlife_db_folder}/{file}.csv',
                    names=column_names,
                    header=0,
                    index_col=False) for file in file_names
    ])
    return database_df


def search_entity_type_in_faithlife_map_with_entity_label(
    entity_label: str,
    faithlife_map: dict[str, str],
) -> str:
    '''Searches for entity type in faithlife database.

    Args:
        ner_label_from_html: 
        Ex: [0, 24, 'The Renaissance in Europe', 'bk.tle%3arenaissanceeurope']
        faithlife_map: 

    Returns:
        The entity type found in faithlife map or 'NONE' if not found.
    '''

    if entity_label in faithlife_map:
        return faithlife_map[entity_label]
    return "NONE"


def extract_name(input_string):
    # Find the index of the opening parenthesis
    start_index = input_string.find('(')
    if start_index != -1:
        # If there is an opening parenthesis, remove it and everything after it
        return input_string[:start_index - 1]
    else:
        # If there are no parentheses, return the entire string
        return input_string


def search_entity_in_faithlife_map(
    sentence: List[str],
    labeled_ner_tuples: List[NERTuple],
    database_df: pd.DataFrame,
    all_entities_in_csv: Set[str],
) -> List[NERTuple]:
    '''
    TODO: Convert lookup into a map later

    Need to clean up labeled_ner_tuples for comparison to work properly
    
    '''
    sentence = sentence.lower()  #fix
    print('sentence: ', sentence)
    print('labeled_ner_tuples: ', labeled_ner_tuples)
    # print('all_entities_in_csv: ', all_entities_in_csv)

    already_labeled_entity_labels = set()
    for ner_tuple in labeled_ner_tuples:
        cleaned_label = replace_special_chars_in_entity_annotations(
            ner_tuple.entity_label)
        already_labeled_entity_labels.add(cleaned_label)

    # print('already_labeled_entity_labels:', already_labeled_entity_labels)
    entities = []
    # Can't locate person because it add labels like "Denis Diderot (French Encyclopaedist)"
    # for bk.%DenisDiderot_Person
    for entity_label in all_entities_in_csv:
        if entity_label not in already_labeled_entity_labels:
            entity_name = database_df.loc[database_df['primary'] ==
                                          entity_label]['label_en'].to_string(
                                              index=False)
            entity_name_lower = entity_name.lower()
            if '(' in entity_name:
                entity_name = extract_name(entity_name)

            entity_type = database_df.loc[database_df['primary'] ==
                                          entity_label]['kind'].to_string(
                                              index=False)
            # print('entity_name:', entity_name)
            # print('entity_label: ', entity_label)
            # print('entity_type: ', entity_type)

            start_index = sentence.find(entity_name_lower)
            if start_index != -1:
                end_index = start_index + len(entity_name_lower)
                entities.append(
                    NERTuple(start_index, end_index, entity_type, entity_label,
                             entity_name))
                # entities.append([
                #     start_index, end_index, entity_type, entity_label,
                #     entity_name
                # ])

            #     print(
            #         f"did find: entity_name: {entity_name} \n sentence:{sentence}"
            #     )
            # else:
            #     print(
            #         f'did not find: entity_name: {entity_name} \n sentence:{sentence}'
            #     )
    print('found_entities:', entities)
    return entities


def make_faithlife_map(faithlife_df: pd.DataFrame) -> Mapping[str, str]:
    '''Convert faithlife dataframe into a hashmap.

    Args:
        faithlife_df: Dataframe containing faithlife concept 
        database data.
    
    Returns:
        A hashmap of entity label to the entity type
        Example:
        { 
            bk.#24Elders:  person
            lsto.AbrahamicCovenant: concept
        }
    '''
    faithlife_map = {}
    for entity_label, entity_type in zip(faithlife_df['primary'],
                                         faithlife_df['kind']):
        faithlife_map[entity_label] = entity_type
    return faithlife_map


def label_events(ner_tuples: List[List[NERTuple]],
                 article_identifer: str) -> List[List[NERTuple]]:
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


def parse_chicago_citation(citation):
    """
    Parses a Chicago style citation in the format "Article Title" in Book Title (Book Author).

    Parameters:
    citation (str): The Chicago style citation string to be parsed.

    Returns:
    dict: A dictionary containing the extracted information, with keys for the article title, book title,
    and book author.
    """
    pattern = r'"(.*)" in (.*) \((.*)\)'
    match = re.match(pattern, citation)
    if not match:
        raise ValueError('Invalid citation format')
    article_title, book_title, book_author = match.groups()
    return {
        'article_title': article_title.strip(),
        'book_title': book_title.strip(),
        'book_author': book_author.strip()
    }