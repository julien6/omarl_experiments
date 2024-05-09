import itertools
import random
from typing import List, Tuple, Union
from organizational_model import cardinality
import re


class token(str):

    def __str__(self) -> str:
        return str(self)

    def __repr__(self) -> str:
        return str(self)


class history_pattern:

    def __init__(self, data: Union[List[token], List['history_pattern']], cardinality: cardinality) -> None:
        self.data = data
        self.cardinality = cardinality

    def __str__(self) -> str:

        def render_aux(seq: history_pattern, num_seq: int) -> str:
            if (len(seq.data) == 0):
                return f's0=([],(1,1))'
            if type(seq.data[0]) == str:
                return f"s{num_seq}=({seq.data},({seq.cardinality.lower_bound}, {seq.cardinality.upper_bound}))"
            else:
                rendered_seqs = []
                for s in seq.data:
                    num_seq += 1
                    rendered_seqs += [render_aux(s, num_seq)]
                return f"s{num_seq + 1}=([" + ",".join(rendered_seqs) + f"], ({seq.cardinality.lower_bound}, {seq.cardinality.upper_bound}))"

        return render_aux(self, -1)


# s1 = history_pattern(["0", "1", "2", "3"], cardinality(2, 2))
# s2 = history_pattern(["6", "3", "9"], cardinality(1, 1))
# s3 = history_pattern([s1, s2], cardinality(1, 1))

# print(s3)


def read_history_patternd_tokens(history_patternd_tokens: history_pattern) -> List[token]:

    def render_aux(seq: history_pattern, num_seq: int) -> List[token]:
        if type(seq.data[0]) == str:
            return seq.data * (seq.cardinality.lower_bound + random.randint(0, seq.cardinality.upper_bound - seq.cardinality.lower_bound))
        else:
            rendered_seqs = []
            for s in seq.data:
                num_seq += 1
                rendered_seqs += [render_aux(s, num_seq)]
            return rendered_seqs * (seq.cardinality.lower_bound + random.randint(0, seq.cardinality.upper_bound - seq.cardinality.lower_bound))

    return list(itertools.chain.from_iterable(render_aux(history_patternd_tokens, -1)))


# print(read_history_patternd_tokens(s3))


def incr_card_history_patternd_tokens(history_patternd_tokens: history_pattern, seq_to_update: List[token]) -> List[token]:

    def incr_aux(seq: history_pattern, num_seq: int) -> List[token]:
        if type(seq.data[0]) == str:
            sq = seq.data * (seq.cardinality.lower_bound + random.randint(0,
                             seq.cardinality.upper_bound - seq.cardinality.lower_bound))
            if sq == seq_to_update:
                seq.cardinality.lower_bound += 1
                seq.cardinality.upper_bound += 1
            return sq
        else:
            rendered_seqs = []
            for s in seq.data:
                num_seq += 1
                rendered_seqs += [incr_aux(s, num_seq)]
            sq = rendered_seqs * (seq.cardinality.lower_bound + random.randint(
                0, seq.cardinality.upper_bound - seq.cardinality.lower_bound))
            if list(itertools.chain.from_iterable(sq)) == seq_to_update:
                seq.cardinality.lower_bound += 1
                seq.cardinality.upper_bound += 1
            return sq

    return list(itertools.chain.from_iterable(incr_aux(history_patternd_tokens, -1)))


def is_subhistory_pattern(b, l):
    j = 0
    ss = []

    i = 0
    while i < len(b):
        if j == len(l) - 1:
            return (True, i-1)
        if l[j] == b[i]:
            ss += [b[j]]
            j += 1
        else:
            if j > 0:
                i -= 1
            ss = []
            j = 0
        i += 1
    return (False, 0)


def parse_tokens(tokens: List[token]) -> history_pattern:

    tmp_seq = []
    history_patternd = history_pattern([], cardinality(1, 1))
    last_added_seq_index = 0
    for t in tokens:
        tmp_seq += [t]
        print(tmp_seq, tokens[0:last_added_seq_index + 1])
        is_ss, ss_index = is_subhistory_pattern(
            tokens[0:last_added_seq_index + 1], tmp_seq)
        if is_ss:
            print(tmp_seq, ss_index, last_added_seq_index)
            if ss_index == last_added_seq_index:
                incr_card_history_patternd_tokens(history_patternd, tmp_seq)
                last_added_seq_index += len(tmp_seq)
        else:
            history_patternd.data += tmp_seq
            last_added_seq_index += len(tmp_seq) - 1
            tmp_seq = []
    return history_patternd


# print(parse_tokens(["1", "2", "3", "1", "2", "3"]))


# print(parse_tokens(read_history_patternd_tokens(s3)))


def eval_str_history_pattern(pattern_string: str):

    pattern_string = pattern_string.replace('#any_seq','[#any_obs,#any_act](1,*)')

    regex = r"(\[[^\]^\[]*\]\(.*?,.*?\))"

    matches = re.findall(regex, pattern_string)

    for group in matches:
        labels, multiplicity = group.split("](")
        labels = ",".join(["\""+lab + "\"" if not "|" in lab else '[' + ",".join(
            ['"'+l+'"' for l in lab.split("|")]) + ']' for lab in labels[1:].split(",")])
        labels = "[" + labels + "]"
        multiplicity = "(" + multiplicity
        multiplicity = multiplicity.replace(
            ",", "\",\"").replace("(", "(\"").replace(")", "\")")
        new_group = '(' + labels + "," + multiplicity + ')'
        pattern_string = pattern_string.replace(group, new_group)

    matches = re.findall(r"(\[.*?\])(\(.*?,.*?\))", pattern_string)

    for group in matches:
        seqs, multiplicity = group
        new_group = '(' + seqs + ',' + multiplicity + ')'
        pattern_string = pattern_string.replace(seqs+multiplicity, new_group)

    pattern_string = pattern_string.replace("))(", ")),(")

    regex = r"(\([0-9\*]*?,[0-9\*]*?\))"

    matches = re.findall(regex, pattern_string)

    for group in matches:
        gr = group.replace("(", "(\"").replace(")", "\")").replace(",", "\",\"")
        pattern_string = pattern_string.replace(group, gr)

    return eval(pattern_string)


def parse_str_history_pattern(pattern_string: str) -> history_pattern:

    eval_str_seq = eval_str_history_pattern(pattern_string)

    print(eval_str_seq)

    def parse_str_history_pattern_aux(eval_seq: Tuple) -> history_pattern:
        values = eval_seq[0]
        is_all_labels = True
        for v in values:
            if type(v) != str:
                is_all_labels = False
                break
        if is_all_labels:
            return history_pattern(values, cardinality(eval_seq[1][0], eval_seq[1][1]))
        else:
            seq = []
            for v in values:
                seq += [parse_str_history_pattern_aux(v)]
            return history_pattern(seq, cardinality(eval_seq[1][0], eval_seq[1][1]))

    return parse_str_history_pattern_aux(eval_str_seq)


# str_seq = "[[o0,a1,o2](1,2),[o2,a2,o3](1,*),[o3,a4,o4](1,1)](1,2)"


print(eval_str_history_pattern('[#any_seq[o01,a01](0,*)#any_seq](1,*)'))


# print(eval_str_history_pattern(str_seq))
