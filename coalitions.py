from itertools import chain, combinations, permutations, combinations_with_replacement
from math import factorial
import re

def get_mcn_candidates(iterable, agents):
    "powerset([1,2,3]) --> () (1,) (2,) (3,) (1,2) (1,3) (2,3) (1,2,3)"
    n = len(agents)
    s = list(iterable)
    return chain.from_iterable(permutations(list([i + 1 for i in range(n)]), r) for r in range(n,len(s)+1))

def get_utility_combinations(agents, singleton_utilities, sum_utils):
    "powerset([1,2,3]) --> () (1,) (2,) (3,) (1,2) (1,3) (2,3) (1,2,3)"
    utility_values = list(singleton_utilities.values())
    n = len(agents)
    all_utility_combinations = []
    for i in range(sum_utils + 1):
        for j in range(sum_utils + 1 - i):
            all_utility_combinations.append((i, j, sum_utils + 1 - i - j - 1))
    #all_utility_combinations = list(chain.from_iterable(combinations_with_replacement(list([i for i in range(sum_utils + 1)]), r) for r in range(n, n+1)))
    #print(all_utility_combinations)
    in_core_candidates = list(); out_of_core_candidates = list();
    for i in range(len(all_utility_combinations)):
        if sum(all_utility_combinations[i]) == sum_utils:
            for j in range(n):
                if all_utility_combinations[i][j] >= utility_values[j]:
                    if j == n - 1:
                        #print(all_utility_combinations[i])
                        in_core_candidates.append((all_utility_combinations[i]))
                    continue
                out_of_core_candidates.append((all_utility_combinations[i]))
                break
    if len(in_core_candidates) == 0:
        print('Empty core')
    else:
        print('Non-empty core')
        print('Core length:', len(in_core_candidates))
        print('Core candidates:')
        print(in_core_candidates)

    print('Non-core length:', len(out_of_core_candidates))
    print('Out-of-core candidates:', out_of_core_candidates)


def estimate_core_values(mapping, agents):
    singletons = list(agents)
    print(singletons)
    singleton_utilities = {}
    for ag in singletons:
        singleton_utilities[ag] = mapping[frozenset([ag])]

    #print(singleton_utilities)
    get_utility_combinations(agents, singleton_utilities, mapping[agents])

def shapley_brute_force(mapping, agents):
    mcn = {}
    shapley_values = {}
    n = len(agents)
    for i in range(n):
        mcn[i+1] = 0
        shapley_values[i+1] = 0

    #print(mcn)
    denom = factorial(n)
    mcn_candidates = list(get_mcn_candidates(mapping, agents))
    grand_coalition_utility = mapping[agents]
    for comb in mcn_candidates:
        comb = list(comb)
        while len(comb) != 0:
            cur_agent = comb.pop()
            if cur_agent in list(agents):
                if len(comb) == 0:
                    mu_cur = grand_coalition_utility
                    mcn[cur_agent] += mu_cur
                    grand_coalition_utility = mapping[agents]
                    break
                mu_cur = grand_coalition_utility - mapping[frozenset(comb)]
                mcn[cur_agent] += mu_cur
                grand_coalition_utility = mapping[frozenset(comb)]

    for k, v in mcn.items():
        sv = float(v / denom)
        shapley_values[k] = sv

    print('Brute force Shapley values:')
    print(shapley_values)
    print('Marginal contribution sums per agent:')
    print(mcn)

def shapley_simple_mcnet(mapping, agents):
    shapley_values = {}
    n = len(agents)


    for cur_agent in list(agents):
        #print(cur_agent)
        shapley_values[cur_agent] = 0
        for pattern, value in mapping.items():
            pattern_str = str.split(str(pattern), '\'')[1]
            #print(pattern_str)
            if cur_agent in pattern_str:
                if len(pattern_str) == 1:
                    # simple one agent rule
                    shapley_values[cur_agent] += value
                if '^' in pattern_str:
                    res_pattern = str.split(pattern_str, '^')
                    #print(res_pattern)
                    if any('~' in element for element in res_pattern):
                        # rule with positive and negative literals
                        current_literal = ''
                        count_p = 0
                        count_n = 0
                        for literal in res_pattern:
                            if '~' in literal:
                                if cur_agent in literal:
                                    current_literal = '~' + str(cur_agent)
                                count_n = count_n + 1
                            else:
                                if cur_agent in literal:
                                    current_literal = cur_agent
                                count_p = count_p + 1

                        if '~' in current_literal:
                            psi = float((factorial(count_p) * factorial(count_n - 1) / factorial(count_p + count_n)) * (-value))
                        else:
                            psi = float((factorial(count_p - 1) * factorial(count_n) / factorial(count_p + count_n)) * value)
                        shapley_values[cur_agent] += psi

                    else:
                        # rule with only positive literals
                        for literal in res_pattern:
                            if literal == cur_agent:
                                shapley_values[cur_agent] += float(value / len(res_pattern))

    print('MCNet Shapley values:', shapley_values)











agents=frozenset([1,2,3])
mapping={
  frozenset([1]):12,
  frozenset([2]):18,
  frozenset([3]):6,
  frozenset([1,2]):60,
  frozenset([1,3]):72,
  frozenset([2,3]):48,
  frozenset([1,2,3]):120
}

agents_mcnet=frozenset(['a', 'b', 'c', 'd'])
mapping_mcnet={
  frozenset(['a^c^~b']):8,
  frozenset(['b^~a']):5,
  frozenset(['c^~a']):2,
  frozenset(['c']):5,
  frozenset(['b^~c']):3,
  frozenset(['d']):9,
  frozenset(['d^c']):4
}

#print(list(get_mcn_candidates(mapping, agents)))
estimate_core_values(mapping, agents)
print()
shapley_brute_force(mapping, agents)
print()
shapley_simple_mcnet(mapping_mcnet, agents_mcnet)