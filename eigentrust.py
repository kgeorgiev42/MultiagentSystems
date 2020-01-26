import random
import networkx as nx
import numpy as np
import math

av_regret_list = []
std_regret_list = []


class Agent:
    def __init__(self, competence):
        self.competence = competence
        self.neighbours = set()
        self.score = [0, 0]

    def do_task(self):
        return random.random() < self.competence

    def sample_dist(self, mu):
        return random.random() <= mu

    def get_target_trust_value(self, target, peer_list):
        sat = target['alpha']
        unsat = target['beta']
        target_S = sat - unsat
        target_S = max(target_S, 0)
        cum_S = 0

        for i in range(len(peer_list)):
            sat_i = peer_list[i]['alpha']
            unsat_i = peer_list[i]['beta']
            S_i = sat_i - unsat_i
            S_i = max(S_i, 0)
            cum_S += S_i

        if cum_S != 0:
            norm_S = float(target_S / cum_S)
            return norm_S
        return 0

    def get_induced_trust_value(self, targetY, peer_listY, targetZ, peer_listZ):
        cum_XZ = 0
        for i in range(min(len(peer_listZ), len(peer_listY))):
            cum_SY = self.get_target_trust_value(peer_listY[i], peer_listY)
            cum_SZ = self.get_target_trust_value(peer_listZ[i], peer_listZ)
            cum_XZ += cum_SY * cum_SZ
        return cum_XZ

    def get_induced_trust_value_v2(self, targetY, peer_listY, targetZ, peer_listZ):
        cum_SY = self.get_target_trust_value(targetY, peer_listY)
        cum_SZ = self.get_target_trust_value(targetZ, peer_listZ)
        cum_XZ = cum_SY * cum_SZ
        return cum_XZ

    def compute_eigentrust_simple(self, C, trust_array, alpha=0, epsilon=0.001):

        m = C.shape[0]
        t = np.array([float(1 / m) for i in range(m)])

        # initialize untrusted pairs with 0
        for i in range(len(trust_array)):
            if trust_array[i] == 0:
                t[i] = 0

        print(t)
        p = t
        diff = 100000
        k = 0

        while diff > epsilon:
            t_k = (1 - alpha) * np.dot(C.T, t) + alpha * p
            diff = np.sum(np.abs(np.subtract(t_k, t)), axis=0)
            print('Iteration: {}, Distance: {}'.format(k + 1, diff))
            t = t_k
            k += 1

        return t


class Environment(nx.DiGraph):
    def add_agents(self, agents):
        m = {}
        i = 0
        for a in agents:
            m[i] = a
            i += 1
        nx.relabel_nodes(self, m, copy=False)

        for n in self.nodes:
            n.neighbours = set(nx.neighbors(self, n))

    def tick(self):
        score = [0, 0]
        for n in self.nodes:
            if self.delegate():
                score[0] += 1
            else:
                score[1] += 1
        return score

    ########################RUN THE EXPERIMENT###########################


NUMAGENTS = 10  # Number of agents

random.seed(0)  # set the random seed
competences = [0.7, 0.9, 0.76, 0.82, 0.99, 0.3, 0.9, 0.1, 0.2, 0.34]

a = []
for i in range(0, NUMAGENTS):
    # a.append(Agent(random.random()))
    a.append(Agent(competences[i]))

G = nx.DiGraph()
G.add_edges_from([(0, 1), (2, 1),
                  (1, 3), (4, 3),
                  (5, 0), (6, 1),
                  (7, 2), (8, 4),
                  (9, 0), (6, 8),
                  (9, 2), (7, 3),
                  (6, 0), (3, 2),
                  (4, 5), (5, 9)])
G = nx.Graph(G)
# G=nx.complete_graph(NUMAGENTS)
# G=nx.path_graph(NUMAGENTS) #create a complete graph, see https://networkx.github.io/documentation/stable/reference/generators.html for other generators
E = Environment(G)
E.add_agents(a)

### reputation
agX_interactions = {
    'alpha': 8,
    'beta': 4
}
agY_interactions = {
    'alpha': 6,
    'beta': 5
}
agZ_interactions = {
    'alpha': 1,
    'beta': 10
}

print('Agent T\'s trust value for X, given X,Y,Z:',
      a[0].get_target_trust_value(
          agX_interactions,
          [agX_interactions, agY_interactions, agZ_interactions]))

print('Agent T\'s trust value for Y, given X,Y,Z:',
      a[0].get_target_trust_value(
          agY_interactions,
          [agX_interactions, agY_interactions, agZ_interactions]))

print('Agent T\'s trust value for Z, given X,Y,Z:',
      a[0].get_target_trust_value(
          agZ_interactions,
          [agX_interactions, agY_interactions, agZ_interactions]))


# how much X trusts Y
agAY_interactions = {
    'alpha': 6,
    'beta': 4
}
agBY_interactions = {
    'alpha': 12,
    'beta': 4
}
agXY_interactions = {
    'alpha': 10,
    'beta': 1
}

#how much Y trusts Z
agYZ_interactions = {
    'alpha': 11,
    'beta': 2
}
agAZ_interactions = {
    'alpha': 1,
    'beta': 4
}
agBZ_interactions = {
    'alpha': 8,
    'beta': 0
}


print('Agents X->Y->Z.. X\'s induced trust value for Z',
      a[0].get_induced_trust_value(
          agXY_interactions,
          [agXY_interactions, agAY_interactions, agBY_interactions],
          agYZ_interactions,
          [agYZ_interactions, agAZ_interactions, agBZ_interactions]))

C = np.array([[0, 0.75, 0.25],
             [0.1, 0.1, 0.8],
             [0.333, 0.333, 0.333]])

trust_array = np.array([1, 1, 1])

C_x = np.array([[0, 1, 0],
                [0.6666667, 0, 0.33333],
                [0.125, 0.875, 0]])

print('C\'s eigentrust values:', a[0].compute_eigentrust_simple(C, trust_array))
print()
print('C\'s eigentrust values:', a[0].compute_eigentrust_simple(C_x, trust_array))

'''
p = np.array([0.33333, 0.33333, 0.33333])

t_x1 = 0.5 * np.dot(C_x.T, p) + (0.5 * p)
print(t_x1)
t_x2 = 0.5 * np.dot(C_x.T, t_x1) + (0.5 * p)
print(t_x2)
t_x3 = 0.5 * np.dot(C_x.T, t_x2) + (0.5 * p)
print(t_x3)
t_x4 = 0.5 * np.dot(C_x.T, t_x3) + (0.5 * p)
print(t_x4)
t_x5 = 0.5 * np.dot(C_x.T, t_x4) + (0.5 * p)
print(t_x5)
t_x6 = 0.5 * np.dot(C_x.T, t_x5) + (0.5 * p)
print(t_x6)
'''