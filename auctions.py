from itertools import chain, permutations, combinations
from copy import deepcopy
from string import ascii_lowercase
import operator
import numpy as np


def powerset(iterable, n=1):
    "powerset([1,2,3]) --> () (1,) (2,) (3,) (1,2) (1,3) (2,3) (1,2,3)"
    s = list(iterable)
    return list(chain.from_iterable(combinations(s, r) for r in range(n, len(s)+1)))

def intersection(lst1, lst2):
    return list(set(lst1) & set(lst2))



def slice_by_lengths(lengths, the_list):
    for length in lengths:
        new = []
        for i in range(length):
            new.append(the_list.pop(0))
        yield new

def return_partition(my_list,num_groups):
    filtered=[]
    for perm in permutations(my_list,len(my_list)):
        for sub_group_perm in subgroups(list(perm)):
            if len(sub_group_perm)==num_groups:
                #sort  within each partition
                sort1=[sorted(i) for i in sub_group_perm]
                #sort by first element of each partition
                sort2=sorted(sort1, key=lambda t:t[0])
                #sort by the number of elements in each partition
                sort3=sorted(sort2, key=lambda t:len(t))
                #if this new sorted set of partitions has not been added, add it
                if sort3 not in filtered:
                    filtered.append(sort3)
    return filtered

def partition(number):
    return {(x,) + y for x in range(1, number) for y in partition(number-x)} | {(number,)}

def subgroups(my_list):
    partitions = partition(len(my_list))
    permed = []
    for each_partition in partitions:
        permed.append(set(permutations(each_partition, len(each_partition))))

    for each_tuple in chain(*permed):
        yield list(slice_by_lengths(each_tuple, deepcopy(my_list)))

def estimate_utilities(agents, items, xor_mappings):
    item_combinations = powerset(items)
    util_dict = {}

    for i in range(1, len(agents) + 1):
        for item in item_combinations:
            # check for exact match
            if len(item) > 1:
                item_str = '^'.join(sorted(item))
            else:
                item_str = ''.join(item)
            util_idx = '{}_{}'.format(item_str, str(i))
            #print(util_idx)
            if item_str in xor_mappings[i - 1]:
                util_dict[util_idx] = int(xor_mappings[i - 1][item_str])
            # if no match for one item, add 0
            elif len(item_str) == 1:
                util_dict[util_idx] = 0
            else:
                # more than one item
                best_value = 0
                item_list = list(item)
                for pattern, value in xor_mappings[i - 1].items():
                    pattern_items = str.split(pattern, '^')
                    if len(pattern_items) < len(item_list):
                        intersection_items = intersection(item_list, pattern_items)
                        if len(intersection_items) > 0 and len(set(item_list) - set(pattern_items)) > 0:
                            if value > best_value:
                                best_value = value

                util_dict[util_idx] = best_value

    print_util_dict(util_dict, agents)

def determine_winners(agents, items, xor_mappings):
    agent_list = list(agents)
    split_combinations = []
    for i in range(1, len(items) + 1):
        for j in return_partition(agent_list, i):
            split_combinations.append(j)

    print(split_combinations)
    d = dict(enumerate(ascii_lowercase))
    dist_dict = {}
    for i in range(len(split_combinations)):
        for j in range(len(split_combinations[i])):
            for k in range(len(split_combinations[i][j])):
                split_combinations[i][j][k] = d[split_combinations[i][j][k] - 1]
        dist_dict[str(split_combinations[i])] = []
        for l in range(1, len(agents) + 1):
            dist_dict[str(split_combinations[i])].append(0)

    print(split_combinations)
    print(dist_dict)

    max_value = 0
    best_key = ''
    for mapping in xor_mappings:
        mapping_best_key = max(mapping.items(), key=operator.itemgetter(1))[0]
        if mapping[mapping_best_key] > max_value:
            max_value = mapping[mapping_best_key]
            best_key = mapping_best_key

    print(best_key)









def print_util_dict(util_dict, agents):
    for i in range(1, len(agents) + 1):
        print('--- Agent ', i, ' characteristic functions ---')
        for pattern, value in util_dict.items():
            cur_agent_items = str.split(pattern, '_')
            #print(cur_agent_items)
            if int(cur_agent_items[1]) == i:
                print('v({})={}'.format(cur_agent_items[0], value))







agents = frozenset([1,2,3])
items = frozenset(['a', 'b', 'c'])
xor_mapping_ag1={
  'a':4,
  'c':2,
  'a^b':7,
  'a^b^c':8,
}
xor_mapping_ag2={
    'b':1,
    'c':5,
    'a^b':10,
    'b^c':17,
}
xor_mapping_ag3={
    'a':1,
    'c':3,
    'a^b':4,
    'a^b^c':14
}
xor_mappings = list([xor_mapping_ag1, xor_mapping_ag2, xor_mapping_ag3])

estimate_utilities(agents, items, xor_mappings)
print()
#determine_winners(agents, items, xor_mappings)


