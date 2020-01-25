import random
import networkx as nx
import matplotlib.pyplot as plt
import time
import numpy as np

av_regret_list = []
std_regret_list = []

class Agent:
  def __init__(self,competence):
    self.competence=competence
    self.neighbours=set()
    self.score=[0,0]

  def do_task(self):
      return random.random()<self.competence

  def sample_dist(self, mu):
      return random.random() <= mu

  def delegate_egreedy(self, epsilon=0.1):
    success=False
    if self.pick_partner_egreedy(epsilon):
            self.score[0]+=1
            success=True
    else:
            self.score[1]+=1          
    return success

  def delegate_ucb(self, alpha=4):
    success=False
    if self.pick_partner_ucb(alpha):
            self.score[0]+=1
            success=True
    else:
            self.score[1]+=1
    return success


  
  def get_reputation(self,target):
    """returns the reputation value for agent target according to this agent"""
    pass

  def pick_partner_ucb(self, alpha=4):
    """select a partner for interaction from the agent's neighbours"""

    mu1 = self.competence
    mu1_cur = 0
    mu2_cur = 0
    t1_cur = 1
    t2_cur = 1
    total_reward = 0
    total_best_reward = 0
    regret = []
    global av_regret
    global std_regret_list
    global NITERATIONS
    t = NITERATIONS

    for neighbour in self.neighbours:
        mu2 = neighbour.competence
        mu1_cur = int(self.sample_dist(mu1))
        mu2_cur = int(self.sample_dist(mu2))

        mu1_ucb = mu1_cur + np.sqrt(alpha * np.log(t + 1) * 1.0 / (2 * t1_cur))
        mu2_ucb = mu2_cur + np.sqrt(alpha * np.log(t + 1) * 1.0 / (2 * t2_cur))

        if mu1_ucb > mu2_ucb:
            new_sample = self.sample_dist(mu1)
            mu1_cur = (t1_cur * mu1_cur + new_sample) * 1.0 / (t1_cur + 1)
            t1_cur += 1
            total_best_reward += new_sample
        else:
            new_sample = self.sample_dist(mu2)
            mu2_cur = (t2_cur * mu2_cur + new_sample) * 1.0 / (t2_cur + 1)
            t2_cur += 1
            total_best_reward += self.sample_dist(mu1)

        total_reward += new_sample
        regret.append(total_best_reward - (total_reward * 1.0))

    av_regret = np.average(regret)
    print('-------------------')
    print('Average observed reward:', float(total_reward / len(self.neighbours)))
    print('Average regret:', av_regret)
    print('Current MU1: ', mu1_cur)
    print('Current MU2: ', mu2_cur)
    print('-------------------')
    av_regret_list.append(av_regret)
    std_regret_list.append(np.std(regret)/2)
    best_mu = max(mu1_cur, mu2_cur)
    return self.sample_dist(best_mu)
    
  def pick_partner_egreedy(self, epsilon=0.1):
    """select a partner for interaction from the agent's neighbours"""

    mu1 = self.competence
    mu1_cur = 0
    mu2_cur = 0
    t1_cur = 0
    t2_cur = 0
    total_reward = 0
    total_best_reward = 0
    regret = []
    global av_regret
    global std_regret_list

    for neighbour in self.neighbours:
        mu2 = neighbour.competence
        explore = self.sample_dist(epsilon)
        if explore == True:
            if mu1_cur < mu2_cur:
                new_sample = self.sample_dist(mu1)
                mu1_cur = (mu1_cur * t1_cur + new_sample) * 1.0 / (t1_cur + 1)
                t1_cur += 1
                total_best_reward += new_sample
            else:
                new_sample = self.sample_dist(mu2)
                mu2_cur = (mu2_cur * t2_cur + new_sample) * 1.0 / (t2_cur + 1)
                t2_cur += 1
                total_best_reward += self.sample_dist(mu1)
        else:
            if mu1_cur > mu2_cur:
                new_sample = self.sample_dist(mu1)
                mu1_cur = (mu1_cur * t1_cur + new_sample) * 1.0 / (t1_cur + 1)
                t1_cur += 1
                total_best_reward += new_sample
            else:
                new_sample = self.sample_dist(mu2)
                mu2_cur = (mu2_cur * t2_cur + new_sample) * 1.0 / (t2_cur + 1)
                t2_cur += 1
                total_best_reward += self.sample_dist(mu1)

        total_reward += new_sample
        regret.append(total_best_reward - (total_reward * 1.0))

    av_regret = np.average(regret)
    print('-------------------')
    print('Average observed reward:', float(total_reward / len(self.neighbours)))
    print('Average regret:', av_regret)
    print('Current MU1: ', mu1_cur)
    print('Current MU2: ', mu2_cur)
    print('-------------------')
    av_regret_list.append(av_regret)
    std_regret_list.append(np.std(regret)/2)
    best_mu = max(mu1_cur, mu2_cur)
    return self.sample_dist(best_mu)



class Environment(nx.DiGraph):
  def add_agents(self,agents):
    m={}
    i=0
    for a in agents:
            m[i]=a
            i+=1
    nx.relabel_nodes(self,m,copy=False)
    
    for n in self.nodes:
            n.neighbours=set(nx.neighbors(self,n))

  def tick(self):
    score=[0,0]
    for n in self.nodes:
            if self.delegate():
                    score[0]+=1
            else:
                    score[1]+=1        
    return score                

########################RUN THE EXPERIMENT###########################
NUMAGENTS=10 #Number of agents
NITERATIONS=100

random.seed(0) #set the random seed
competences = [0.7, 0.9, 0.76, 0.82, 0.99, 0.3, 0.9, 0.1, 0.2, 0.34]

a=[]
for i in range(0,NUMAGENTS):
        #a.append(Agent(random.random()))
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
#G=nx.complete_graph(NUMAGENTS)
#G=nx.path_graph(NUMAGENTS) #create a complete graph, see https://networkx.github.io/documentation/stable/reference/generators.html for other generators
E=Environment(G)
E.add_agents(a)

std_list = []
sum_pos = 0
sum_neg = 0
#random.seed(time.time()) #uncomment if you want different experiments on same graph
for i in range(NITERATIONS): #run for 100 rounds
        score = [0,0]
        for a in E.nodes:
            s=a.delegate_ucb(alpha=4)
            if s:
              score[0] +=1
            else:
              score[1] +=1
        std_list.append(np.std(score))
        print('Round {}: (Positive: {}, Negative: {})'.format(i + 1, score[0], score[1]))
        sum_pos += score[0]; sum_neg += score[1];

#sp = dict(nx.all_pairs_shortest_path(G))
#print(sp[9])
nx.draw(G, with_labels=True, font_weight='bold')
plt.show()

print('Average regret value:', np.average(av_regret_list))
print('Total sum over rounds: (Positive: {}, Negative: {})'.format(sum_pos, sum_neg))
print('Positive interaction ratio:', round((sum_pos / (sum_pos + sum_neg)), 2))
print('Interaction distributions:')

plt.plot(std_list, c='green', alpha=0.7, label='std')
plt.xlabel('Iterations')
plt.ylabel('Number of interactions')
plt.legend()
plt.show()

iterations = np.arange(NITERATIONS*NUMAGENTS)
plt.errorbar(iterations, av_regret_list, yerr=std_regret_list, label='Epsilon=0.1')
plt.xlabel('Iterations')
plt.ylabel('Regret')
plt.legend()
plt.show()
