import random
import networkx as nx
import matplotlib.pyplot as plt
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

    def get_reputation_BRS_single(self, target):
        """returns the reputation value for agent target according to this agent"""
        alpha = target['alpha']
        beta = target['beta']

        numerator = math.gamma(alpha + beta + 2)
        denominator = math.gamma(alpha + 1) * math.gamma(beta + 1)
        p = float((alpha + 1) / (alpha + beta + 2))

        phi = float( (numerator / denominator) * math.pow(p, alpha) * math.pow((1 - p), beta))
        rep_rating = float((alpha - beta) / (alpha + beta + 2))
        return phi, rep_rating

    def get_cum_reputation_BRS(self, target, forget_factor=1):
        """returns the reputation value for agent target according to this agent"""
        cum_pos = 0; cum_neg = 0;
        n = len(target)
        for i in range(n):
            if forget_factor == 0:
                cum_pos = target[n - 1]['alpha']
                cum_neg = target[n - 1]['beta']
                break
            cum_pos += target[i]['alpha'] * math.pow(forget_factor, n - i + 1)
            cum_neg += target[i]['beta'] * math.pow(forget_factor, n - i + 1)

        cum_dict = {
            'alpha': cum_pos,
            'beta': cum_neg,
        }

        cum_phi, cum_rep_rating = self.get_reputation_BRS_single(cum_dict)

        return cum_phi, cum_rep_rating

    def get_discounted_opinion(self, target):
        """returns the reputation value for agent target according to this agent"""
        alphaZ = target[2]['alpha']
        alphaY = target[1]['alpha']
        betaZ = target[2]['beta']
        betaY = target[1]['beta']

        bXY = float(alphaY / (alphaY + betaY + 2))
        dXY = float(betaY / (alphaY + betaY + 2))
        uXY = float(2 / (alphaY + betaY + 2))
        bYZ = float(alphaZ / (alphaZ + betaZ + 2))
        dYZ = float(betaZ / (alphaZ + betaZ + 2))
        uYZ = float(2 / (alphaZ + betaZ + 2))

        bXZ = bXY * bYZ
        dXZ = bXY * dYZ
        uXZ = dXY + uXY + bXY * uYZ
        print('Belief: {}, Disbelief: {}, Uncertainty: {}'.format(bXZ, dXZ, uXZ))
        return bXZ, dXZ, uXZ

    def get_discounted_reputation(self, target):

        alphaZ = target[2]['alpha']
        alphaY = target[1]['alpha']
        betaZ = target[2]['beta']
        betaY = target[1]['beta']

        #phi_XY, rating_XY = self.get_reputation_BRS_single(target[1])
        #phi_YZ, rating_YZ = self.get_reputation_BRS_single(target[2])

        alpha_XZ = (2 * alphaY * alphaZ) / (((betaY + 2) * (alphaZ + betaZ + 2)) + (2 * alphaY))
        beta_XZ = (2 * alphaY * betaZ) / (((betaY + 2) * (alphaZ + betaZ + 2)) + (2 * alphaY))

        XZ_dict = {
            'alpha':alpha_XZ,
            'beta':beta_XZ
        }

        print('Induced positive feedback:', alpha_XZ)
        print('Induced negative feedback:', beta_XZ)
        print('Induced reputation rating:', self.get_reputation_BRS_single(XZ_dict))

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
    'alpha': 5,
    'beta': 5
}
agZ_interactions = {
    'alpha': 1,
    'beta': 10
}

agT_interactions = {
    'alpha': 0,
    'beta': 0
}

print('Agent X reputation (trust based): ', a[0].get_reputation_BRS_single(agX_interactions))
print('Agent Y reputation (trust based): ', a[0].get_reputation_BRS_single(agY_interactions))
print('Agent Z reputation (trust based): ', a[0].get_reputation_BRS_single(agZ_interactions))
ag_p_feedback = list(np.arange(0, 10))
ag_n_feedback = list(np.arange(0, 10))


ag_p_list = []; ag_p_norm = [];
ag_n_list = []; ag_n_norm = [];
for i in range(len(ag_p_feedback)):
    agT_interactions['alpha'] = ag_p_feedback[i]
    phi, norm = a[0].get_reputation_BRS_single(agT_interactions)
    ag_p_list.append(phi)
    ag_p_norm.append(norm)

agT_interactions['alpha'] = 0
for i in range(len(ag_n_feedback)):
    agT_interactions['beta'] = ag_n_feedback[i]
    phi, norm = a[0].get_reputation_BRS_single(agT_interactions)
    ag_n_list.append(phi)
    ag_n_norm.append(norm)

plt.scatter(ag_p_list, ag_p_feedback)
plt.show()
plt.plot(ag_p_norm, ag_p_feedback, label='positive')
plt.plot(ag_n_norm, ag_n_feedback, label='negative')
plt.xlabel('Expected normalized reputation')
plt.ylabel('N positive/negative feedbacks')
plt.legend()
plt.show()


print('Agents X,Y,Z combined reputation feedback (full): ', a[0].get_cum_reputation_BRS([agX_interactions, agY_interactions, agZ_interactions]))
print()
print('Agents X->Y->Z.. X\'s induced opinion of Z:')
a[0].get_discounted_opinion([agX_interactions, agY_interactions, agZ_interactions])
print()
print('Agents Y->Z->X.. Y\'s induced opinion of X:')
a[0].get_discounted_opinion([agY_interactions, agZ_interactions, agX_interactions])
print()
print('Agents X->Z->Y.. X\'s induced opinion of Y:')
a[0].get_discounted_opinion([agX_interactions, agZ_interactions, agY_interactions])
print()
print('Agents Y->X->Z.. Y\'s induced opinion of Z:')
a[0].get_discounted_opinion([agY_interactions, agX_interactions, agZ_interactions])
print()

print('Agents X->Y->Z.. X\'s induced reputation rating for Z:')
a[0].get_discounted_reputation([agX_interactions, agY_interactions, agZ_interactions])
print('Agents Y->X->Z.. Y\'s induced reputation rating for Z:')
a[0].get_discounted_reputation([agY_interactions, agX_interactions, agZ_interactions])
print('Agents X->Z->Y.. X\'s induced reputation rating for Y:')
a[0].get_discounted_reputation([agX_interactions, agZ_interactions, agY_interactions])
print('Agents Y->Z->X.. Y\'s induced reputation rating for X:')
a[0].get_discounted_reputation([agY_interactions, agZ_interactions, agX_interactions])