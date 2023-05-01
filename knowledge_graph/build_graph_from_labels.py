# 'concept', 'writing', 'person', 'place', 'artifact', 'thing', 'book'

import csv
import glob
import pandas as pd

from neo4j import GraphDatabase, basic_auth
from typing import List, Mapping


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
                                      index=False).lower()

    entity_type = database_df.loc[database_df['primary'] ==
                                  entity_label]['kind'].to_string(index=False)

    return entity_name, entity_type


def ingest_csv_to_neo4j(database_df, csv_file_path: str, uri: str, user: str,
                        password: str):
    driver = GraphDatabase.driver(uri, auth=basic_auth(user, password))

    with driver.session() as session:
        with open(csv_file_path, 'r') as csv_file:
            csv_reader = csv.DictReader(csv_file)
            for row in csv_reader:
                subject_entity_label = row['subj']
                relation_type = row['rel']
                object_entity_label = row['obj']

                subject_entity_name, subject_entity_type = get_entity_name_and_type_from_map(
                    database_df, subject_entity_label)
                object_entity_name, object_entity_type = get_entity_name_and_type_from_map(
                    database_df, object_entity_label)

                query = f'''
                MERGE (s:{subject_entity_type} {{name: {subject_entity_name}, label: {subject_entity_label}}})
                MERGE (o:{object_entity_type} {{name: {object_entity_name}, label: {object_entity_label}}})
                MERGE (s)-[r:`{relation_type}`]->(o)
                '''

                session.run(query,
                            subject=subject_entity_name,
                            relation_type=relation_type,
                            object=object_entity_name)
    driver.close()


database_df = load_faithlife_database_to_single_df(
    '../preprocessing/faithlife_data/entities')

neo4j_uri = "bolt://localhost:7687"
neo4j_user = "neo4j"
neo4j_password = "faithlife"
article_data = load_relation_data('../preprocessing/faithlife_data')
for csv_files in article_data:
    ingest_csv_to_neo4j(database_df, csv_files, neo4j_uri, neo4j_user,
                        neo4j_password)
