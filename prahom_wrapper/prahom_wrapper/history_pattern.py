import copy
import random
import re

from typing import List, Tuple
from utils import cardinality, history, label, history_pattern_str, history_str


MAX_OCCURENCE = 10


def _parse_into_tuple(string_pattern: str) -> Tuple[List, cardinality]:
    stack = []
    i = 0
    while i < len(string_pattern):
        character = string_pattern[i]
        if character == "[":
            stack += ["["]
        elif character == "]":
            stack[-1] += "]"
            if string_pattern[i+1] == "(":
                card = ""
                i += 1
                while i < len(string_pattern):
                    char_card = string_pattern[i]
                    card += char_card
                    if char_card == ")":
                        break
                    i += 1
                sequence = stack.pop()
                sequence = f'({sequence},{card})'
                if (i == len(string_pattern) - 1):
                    seq = sequence.replace("[", "[\"").replace("]", "\"]") \
                        .replace("(", "(\"").replace(")", "\")").replace(",", "\",\"") \
                        .replace("\"(", "(").replace(")\"", ")") \
                        .replace("\"[", "[").replace("]\"", "]")
                    seq = _uniformize(seq)
                    return eval(seq)
                stack[-1] += sequence
        else:
            stack[-1] += character
        i += 1


def _uniformize(string_pattern: str) -> str:

    # lonely labels in the begining of sequences
    regex = r'\[([\",0-9A-Za-z]{2,}?),\('
    matches = re.findall(regex, string_pattern)

    for group in matches:
        if group != "":
            string_pattern = string_pattern.replace(
                '[' + group + ',(', '[' + f"([{group}],('1','1'))" + ',(')

    # lonely labels in the middle of sequences
    regex = r'\)\),([\",0-9A-Za-z]{2,}?),\('
    matches = re.findall(regex, string_pattern)

    for group in matches:
        if group != "":
            string_pattern = string_pattern.replace(
                ')),' + group + ',(', ')),' + f"([{group}],('1','1'))" + ',(')

    # lonely labels in the end of sequences
    regex = r'\),([\",0-9A-Za-z]{2,}?)\],\('
    matches = re.findall(regex, string_pattern)

    for group in matches:
        if group != "":
            string_pattern = string_pattern.replace(
                '),' + group + '],(', '),' + f"([{group}],('1','1'))" + '],(')

    return string_pattern


def _is_only_labels(tuple_pattern: Tuple[List, cardinality]) -> bool:
    label_or_tuple_list, card = tuple_pattern
    for label_or_tuple in label_or_tuple_list:
        if type(label_or_tuple) == tuple:
            return False
    return True


def _rdm_occ_card(card: cardinality) -> int:
    min_card = card[0]
    if type(min_card) == str:
        min_card = int(min_card)
    max_card = card[1]
    if max_card == "*":
        max_card = random.randint(0, MAX_OCCURENCE)
    elif type(max_card) == str:
        max_card = int(max_card)
    return random.randint(min_card, max_card)


def _sample(pattern_string: str):

    def sample_aux(history_tuples):

        if _is_only_labels(history_tuples):
            seq, card = history_tuples
            # seq = ["o" if label == "#any" else label for label in seq]

            seqs = []
            anonymous_label_index = 0
            for i in range(0, _rdm_occ_card(card)):
                for label in seq:
                    if label == "#any":
                        seqs += [f"o_{anonymous_label_index}"]
                        anonymous_label_index += 1
                    else:
                        seqs += [label]

            return ",".join(seqs)

        else:
            seq, card = history_tuples
            generated_seq = ""
            for sub_seq in seq:
                if generated_seq != "":
                    generated_seq += ","
                generated_seq += sample_aux(sub_seq)
            return ",".join([generated_seq] * _rdm_occ_card(card))

    return sample_aux(_parse_into_tuple(pattern_string)).replace(",,", ",")


def _match(pattern_string, string):

    def match_aux(history_tuples, call_num, length, max_length):

        if _is_only_labels(history_tuples):
            seq, card = history_tuples

            sub_string = ','.join(seq)
            sub_string = sub_string.replace("#any", ".*?")
            card_min = str(card[0])
            card_max = str(card[1])
            if card_max == "*":
                card_max = MAX_OCCURENCE

            pattern = f"({sub_string}|{sub_string},|,{sub_string}|,{sub_string},)" + \
                "{" + str(card_min) + "," + str(card_max) + "}"

            return pattern, length + 1, pattern

        else:
            seq, card = history_tuples

            card_min = int(card[0])
            card_max = card[1]
            last_sequence = None

            sequence_pattern = ""
            if card_max == "*":
                card_max = MAX_OCCURENCE
            else:
                card_max = int(card_max)

            for sub_seq in seq:
                pattern, length, last_sequence = match_aux(
                    sub_seq, call_num+1, length, max_length)
                sequence_pattern += pattern
                if length >= max_length and max_length != -1:
                    sequence_pattern += ".*"
                    break
            if call_num == 0:
                return "(^" + sequence_pattern + "[,]{0,1}$)" + "{" + str(card_min) + "," + str(card_max) + "}", length, last_sequence
            return "(" + sequence_pattern + "[,]{0,1})" + "{" + str(card_min) + "," + str(card_max) + "}", length, last_sequence

    pattern, overall_length, last_sequence = match_aux(
        _parse_into_tuple(pattern_string), 0, 0, -1)

    next_expected_sequence = None
    matches = re.match(pattern, string)
    matched = None
    is_matched = False
    if matches:
        is_matched = True

    length = 0
    for i in range(1, overall_length + 1):
        pattern, length, last_sequence = match_aux(
            _parse_into_tuple(pattern_string), 0, 0, i)
        old_matches = copy.copy(matches)
        matches = re.match(pattern, string)
        matched = None
        if matches:
            matched = list(matches.groups())[0]
        else:
            matched = string
            if old_matches is not None:
                matched = list(old_matches.groups())[0]
            length -= 1
            next_expected_sequence = last_sequence
            break

    if next_expected_sequence:
        seq, card = next_expected_sequence.split("){")
        next_expected_sequence = (
            seq.split("|")[0][1:].split(","), card[:-1].split(","))
        next_expected_sequence = (next_expected_sequence[0], cardinality(
            next_expected_sequence[1][0], next_expected_sequence[1][1]))

    return is_matched, matched, float(length / overall_length), next_expected_sequence


class history_pattern:

    def __init__(self, history_pattern_string: history_pattern_str) -> None:
        self.history_pattern_string = history_pattern_string

    def sample(self) -> str:
        return _sample(self.history_pattern_string)

    def match(self, history_string: history) -> Tuple[bool, str, float, Tuple[List[str], cardinality]]:
        return _match(self.history_pattern_string, history_string)


class history_patterns:

    def __init__(self) -> None:
        self.patterns: List[history_pattern] = []

    def add_pattern(self, history_pattern_string: history_pattern_str) -> None:
        self.patterns += [history_pattern(history_pattern_string)]

    def get_actions(self, history: history_str, observation_label: label) -> List[label]:
        actions = []
        for pattern in self.patterns:
            match, matched, coverage, next_seq = pattern.match(
                history + f',{observation_label}')

            next_action = None
            if next_seq is not None:
                expected_next_sequence = next_seq[0]
                matched_sequence = matched.split(",")

                next_action = None
                for i in range(0, len(expected_next_sequence)):
                    e = expected_next_sequence[:len(expected_next_sequence)-i]
                    m = matched_sequence[len(
                        matched_sequence)-len(e):len(matched_sequence)]
                    if m == e:
                        # print(e, m, i)
                        next_action = expected_next_sequence[len(
                            expected_next_sequence)-i]

                if next_action is None and match:
                    next_action = expected_next_sequence[0]
            if next_action is not None:
                actions += [next_action]
        return actions


if __name__ == '__main__':

    # hist_pattern = history_pattern("[0,1,2,3,[#any](0,*),4,5,6](1,1)")

    # history = "0,1"

    # match, matched, coverage, next_seq = hist_pattern.match(history)

    # print(history, next_seq)

    # print((hist_pattern.sample()))

    hp = history_patterns()
    hp.add_pattern("[0,1,2,3,[#any](0,*),4,5,6](1,1)")
    hp.add_pattern("[0,1,2,7,9](1,1)")
    print(hp.get_actions("0,1,2,3,89,10,4", "5"))
