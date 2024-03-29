# 'concept', 'writing', 'person', 'place', 'artifact', 'thing', 'book'

import csv
import glob
import pandas as pd
import json
import sys
import os

from neo4j import GraphDatabase, basic_auth
from typing import List, Mapping, Tuple

# Get the current directory of the script
current_dir = os.path.dirname(os.path.abspath(__file__))

# Get the parent directory of the current directory
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

from preprocessing.faithlife_utils import make_faithlife_map_2
#import preprocessing path


def load_relation_data(path_to_church_articles: str) -> List[str]:
    annotated_data = glob.glob(
        f'{path_to_church_articles}/church-history-articles-en/*.csv')
    annotated_data.remove(
        f'{path_to_church_articles}/church-history-articles-en/Relations.csv')
    annotated_data.remove(
        f'{path_to_church_articles}/church-history-articles-en/Roles.csv')
    return annotated_data


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


def get_entity_name_and_type_from_map(database_df: pd.DataFrame,
                                      entity_label: str):
    entity_name = database_df.loc[database_df['primary'] ==
                                  entity_label]['label_en'].to_string(
                                      index=False)
    #check if entity_name is an empty panda series
    if entity_name == 'Series([], )':
        entity_name = entity_label.strip('.')[1]

    entity_type = database_df.loc[database_df['primary'] ==
                                  entity_label]['kind'].to_string(index=False)
    if entity_type == 'Series([], )':
        entity_type = 'no_type'

    return entity_name, entity_type


def ingest_csv_to_neo4j(database_df, csv_file_path: str, uri: str, user: str,
                        password: str):
    driver = GraphDatabase.driver(uri, auth=basic_auth(user, password))

    with driver.session() as session:
        with open(csv_file_path, 'r') as csv_file:
            csv_reader = csv.DictReader(csv_file)
            for row in csv_reader:
                subject_entity_label = row['subj'].strip()
                relation_type = row['rel'].strip()
                object_entity_label = row['obj'].strip()

                subject_entity_name, subject_entity_type = get_entity_name_and_type_from_map(
                    database_df, subject_entity_label)
                object_entity_name, object_entity_type = get_entity_name_and_type_from_map(
                    database_df, object_entity_label)

                #query map if entity exists in database
                # otherwise, we can infer entity type from the label and get entity name by indice in sentence

                query = f'''
                MERGE (s:{subject_entity_type} {{name: '{subject_entity_name}', label: '{subject_entity_label}'}})
                MERGE (o:{object_entity_type} {{name: '{object_entity_name}', label: '{object_entity_label}'}})
                MERGE (s)-[r:`{relation_type}`]->(o)
                '''

                session.run(query,
                            subject=subject_entity_name,
                            relation_type=relation_type,
                            object=object_entity_name)
    driver.close()


def lookup_entity_label_in_faithlife_map(entity_name: str):
    if entity_name in faithlife_map:
        return faithlife_map[entity_name][0], faithlife_map[entity_name][1]
    else:
        return 'no_label', 'no_type'


def ingest_predictions_to_neo4j(json_file_path: str, uri: str, user: str,
                                password: str):
    driver = GraphDatabase.driver(uri, auth=basic_auth(user, password))

    with driver.session() as session:
        with open(json_file_path, 'r') as json_file:
            for line in json_file:
                json_data = json.loads(line.strip())
                sentence_index = 0
                for sent_idx, sentence in enumerate(json_data['sentences']):

                    for predicted_entity_tuples in json_data['predicted_ner'][
                            sent_idx]:
                        entity_name_tokenized = sentence[
                            predicted_entity_tuples[0] -
                            sentence_index:predicted_entity_tuples[1] -
                            sentence_index + 1]

                        entity_name = " ".join(entity_name_tokenized).rstrip()
                        if "'" in entity_name:
                            entity_name = entity_name.replace("'", "\\'")

                        entity_label, _ = lookup_entity_label_in_faithlife_map(
                            entity_name)
                        entity_type = predicted_entity_tuples[2]

                        query = f'''
                        MERGE (s:{entity_type} {{name: '{entity_name}', label: '{entity_label}'}})
                        
                        '''

                        session.run(query,
                                    entity_name=entity_name,
                                    entity_type=entity_type,
                                    entity_label=entity_label)

                    for predicted_relation_tuples in json_data[
                            'predicted_relations'][sent_idx]:
                        subj_entity_start_index = predicted_relation_tuples[0]
                        subj_entity_end_index = predicted_relation_tuples[1]

                        subj_entity_name_tokenized = sentence[
                            subj_entity_start_index -
                            sentence_index:subj_entity_end_index -
                            sentence_index + 1]

                        subj_entity_name = " ".join(
                            subj_entity_name_tokenized).rstrip()

                        subj_entity_label, subj_entity_type = lookup_entity_label_in_faithlife_map(
                            subj_entity_name)

                        obj_entity_start_index = predicted_relation_tuples[2]
                        obj_entity_end_index = predicted_relation_tuples[3]

                        obj_entity_name_tokenized = sentence[
                            obj_entity_start_index -
                            sentence_index:obj_entity_end_index -
                            sentence_index + 1]

                        obj_entity_name = " ".join(
                            obj_entity_name_tokenized).rstrip()

                        obj_entity_label, obj_entity_type = lookup_entity_label_in_faithlife_map(
                            obj_entity_name)

                        relation_type = predicted_relation_tuples[4]

                        if "'" in subj_entity_name:
                            subj_entity_name = subj_entity_name.replace(
                                "'", "\\'")
                        if "'" in obj_entity_name:
                            obj_entity_name = obj_entity_name.replace(
                                "'", "\\'")

                        # Create a relationship between subj and obj entities
                        query = f'''
                        MATCH (subj:{subj_entity_type} {{ name: '{subj_entity_name}' }})
                        MATCH (obj:{obj_entity_type} {{ name: '{obj_entity_name}' }})
                        MERGE (subj)-[:`{relation_type}`]->(obj)
                        '''

                        session.run(query)

                    sentence_index += len(sentence)


database_df = load_faithlife_database_to_single_df(
    '../preprocessing/faithlife_data/entities')
faithlife_map = make_faithlife_map_2(database_df)

neo4j_uri = "bolt://localhost:7687"
neo4j_user = "neo4j"
neo4j_password = "faithlife"
article_data = load_relation_data('../preprocessing/faithlife_data')
for csv_files in article_data:
    ingest_csv_to_neo4j(database_df, csv_files, neo4j_uri, neo4j_user,
                        neo4j_password)

ingest_predictions_to_neo4j('../prediction_data/unlabeled_predictions.json',
                            neo4j_uri, neo4j_user, neo4j_password)
