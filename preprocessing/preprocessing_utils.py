import re
import spacy

nlp = spacy.load("en_core_web_sm")

from typing import List


def replace_special_chars_in_entity_annotations(entity_label: str):
    '''Replace special characters in entity annotations in Article.

    Args:
        entity_label: Entity label from html from Article
    Return
        Entity labeled with annotaitons removed
    
    ex: 
    
    bk.tle%3arenaissanceeurope --> bk.tlearenaissanceeurope
    '''
    entity_label = entity_label.replace("%25", "%")
    entity_label = entity_label.replace("%23", "%")
    entity_label = entity_label.replace("%40", "@")  #for places
    entity_label = entity_label.replace("%3", "%")  #for tle
    return entity_label


def tokenize_sentence(sentence: str) -> List[str]:
    '''Tokenize sentence from faithlife article.

    Args:
        sentence: Sentence from faithlife article.
    
    Return
        List of tokens from sentence.
    '''

    # Find all instances of text within parentheses
    parentheses_matches = re.findall(r'\([^()]+\)', sentence)
    # Replace the text within parentheses with placeholder tokens
    for i, match in enumerate(parentheses_matches):
        sentence = sentence.replace(match, f'{{{i}}}')
    # Tokenize the sentence
    tokens = sentence.split()
    # Replace the placeholder tokens with the original text within parentheses
    for i, match in enumerate(parentheses_matches):
        tokens = [token.replace(f'{{{i}}}', match) for token in tokens]

    #deal with period at the end
    if tokens and tokens[-1] == '.':
        punctuation_mark = tokens[-1][-1]
        tokens[-1] = tokens[-1][0:len(tokens[-1]) - 1]
        tokens.append(punctuation_mark)

    tokens = seperate_punctuation_marks(tokens)
    tokens = separate_parentheses(tokens)
    return tokens


def tokenize_entity(entity: str) -> List[str]:
    '''Tokenize sentence from faithlife article.

    Args:
        sentence: Sentence from faithlife article.
    
    Return
        List of tokens from sentence.
    '''

    # Find all instances of text within parentheses
    parentheses_matches = re.findall(r'\([^()]+\)', entity)
    # Replace the text within parentheses with placeholder tokens
    for i, match in enumerate(parentheses_matches):
        entity = entity.replace(match, f'{{{i}}}')
    # Tokenize the sentence
    tokens = entity.split()

    # Replace the placeholder tokens with the original text within parentheses
    for i, match in enumerate(parentheses_matches):
        tokens = [token.replace(f'{{{i}}}', match) for token in tokens]

    tokens = seperate_punctuation_marks(tokens)
    tokens = separate_parentheses(tokens)
    return tokens


def separate_parentheses(input_list):
    output = []
    for word in input_list:

        if word and word[0] == '(' and word[-1] == ')':
            output.append('(')
            output.append(word[1:len(word) - 1])
            output.append(')')
        elif word and word[0] == '(':
            output.append('(')
            output.append(word[1:])
        elif word and word[-1] == ')':
            output.append(word[0:len(word) - 1])
            output.append(')')
        else:
            output.append(word)
    return output


def seperate_punctuation_marks(input_list):
    output = []
    punctuation_marks = {'"', "'", ',', '“', ':', ';'}
    for word in input_list:
        if word and word[0] in punctuation_marks and word[
                -1] in punctuation_marks:
            output.append(word[0])
            output.append(word[1:len(word) - 1])
            output.append(word[-1])

        elif word and word[0] in punctuation_marks:
            output.append(word[0])
            output.append(word[1:])

        elif word and word[-1] in punctuation_marks:
            output.append(word[0:len(word) - 1])
            output.append(word[-1])

        else:
            output.append(word)
    return output