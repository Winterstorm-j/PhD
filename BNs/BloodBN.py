from numpy import matrix
from pgmpy.factors.discrete import TabularCPD
from pgmpy.models import BayesianNetwork
from pgmpy.inference import BeliefPropagation
import networkx as nx
import matplotlib.pyplot as plt
import pygraphviz
import matplotlib.patches as mpatches

DAG = BayesianNetwork()

DAG.add_nodes_from(['Hypotheses','Blood present','Biological Material',
                    'Stain Appearance','Luminol','Luminol Stain intensity','KM','KM Stain intensity','Combur','Combur Stain intensity','Hematrace (KM)',
                    'Hematrace (KM) Stain intensity','Hematrace (Combur)','Hematrace (Combur) Stain intensity','Hp','Hd', 'LR'])

DAG.add_edges_from([('Hypotheses', 'Blood present'), ('Hypotheses', 'Hp'), ('Hypotheses', 'Hd'), ('Hp', 'LR'), ('Hd', 'LR'), 
    ('Blood present', 'Biological Material'), ('Biological Material', 'KM'), ('Biological Material', 'Hematrace (KM)'), ('KM', 'Hematrace (KM)'),
    ('Hematrace (KM)', 'Hematrace (KM) Stain intensity'), ('Biological Material', 'Hematrace (KM) Stain intensity'), ('Biological Material', 'KM Stain intensity'), ('KM', 'KM Stain intensity'),
    ('Biological Material', 'Stain Appearance'), ('KM', 'Stain Appearance'), ('Biological Material', 'Combur'), ('Stain Appearance', 'Combur'),
    ('Biological Material', 'Hematrace (Combur)'), ('Combur', 'Hematrace (Combur)'), ('Biological Material', 'Hematrace (Combur) Stain intensity'), ('Hematrace (Combur)', 'Hematrace (Combur) Stain intensity'),
    ('Biological Material', 'Combur Stain intensity'), ('Combur', 'Combur Stain intensity'), ('Stain Appearance', 'Luminol'), ('Biological Material', 'Luminol'),
    ('Biological Material', 'Luminol Stain intensity'), ('Luminol', 'Luminol Stain intensity')])

c1_cpd = TabularCPD("Hypotheses", 2, [[0.5], [0.5]],
                    state_names = {'Hypotheses':['Hp', 'Hd']}
                    )

c2_cpd = TabularCPD("Blood present", 2, [[1,0],[0, 1]],
                    state_names = {'Blood_present':["yes", "no"]},
                       evidence=['Hypotheses'],
                       evidence_card=[2]
                    )

c3_cpd = TabularCPD("Biological Material", 8, [[1,0],
                                               [0,0.142857],
                                               [0,0.142857],
                                               [0,0.142857],
                                               [0,0.142857],
                                               [0,0.142857],
                                               [0,0.142857],
                                               [0,0.142857]],
                    state_names = {'Biogical_Mat':["Blood", "Semen", "Saliva", "Sweat", "Faeces", "Urine", "Vaginal material", "Breast milk"]},
                       evidence=['Blood present'],
                       evidence_card=[2]
                    )

c4_cpd = TabularCPD("Stain Appearance", 5, [
        [0.02,0.56,0.83,0.65,0.01,0.49,0.49,0.12],
        [0.01,0.22,0.07,0.02,0.01,0.01,0.18,0.11],
        [0.01,0.20,0.03,0.12,0.01,0.50,0.27,0.65],
        [0.93,0.01,0.01,0.01,0.28,0.01,0.06,0.01],
        [0.04,0.01,0.07,0.20,0.67,0.01,0.01,0.12]],
                    state_names = {'Stain_Appearance':["none", "white/crystaline", "white/yellow", "Red/Brown", "other"]},
                       evidence=['Biological Material'],
                       evidence_card=[8]
                    )

c5_cpd = TabularCPD("Luminol", 2, [[0.5,0.5,0.5,0.9865,0.5,
                                    0.5,0.8333,0.1667,0.5,0.5,
                                    0.7344,0.8333,0.5,0.5,0.9286,
                                    0.0385,0.5,0.5,0.5,0.3125,
                                    0.5,0.5,0.5,0.5,0.15,
                                    0.0416,0.5,0.1336,0.5,0.5,
                                    0.1429,0.5,0.1,0.5,0.5,
                                    0.125,0.5,0.5,0.5,0.5],
                                   [0.5,0.5,0.5,0.0135,0.5,
                                    0.5,0.1667,0.8333,0.5,0.5,
                                    0.2656,0.1668,0.5,0.5,0.0714,
                                    0.98615,0.5,0.5,0.5,0.6875,
                                    0.5,0.5,0.5,0.5,0.85,
                                    0.9583,0.5,0.8864,0.5,0.5,
                                    0.8571,0.5,0.9,0.5,0.5,
                                    0.875,0.5,0.5,0.5,0.5]],
                    state_names = {'Luminol':["positive", "negative"]},
                    evidence=['Biological Material',"Stain Appearance"],
                    evidence_card=[8,5]
                    )

c6_cpd = TabularCPD("Luminol Stain intensity", 3, [[0.98,0.01,0.175,0.8,
                                               0.59,0.4,0.19,0.79,
                                               0.17,0.78,0.01,0.89,
                                               0.13,0.84,0.09,0.73], 
                                              [0.01,0.01,0.025,0.025,
                                               0.01,0.01,0.02,0.02,
                                               0.04,0.04,0.10,0.10,
                                               0.03,0.03,0.18,0.18],
                                              [0.01,0.98,0.8,0.175,
                                               0.4,0.59,0.79,0.19,
                                               0.78,0.17,0.89,0.01,
                                               0.84,0.13,0.73,0.09]],
                    state_names = {'C6_StainIntesity':["strong", "weak", "none"]},
                    evidence=['Biological Material',"Luminol"],
                    evidence_card=[8,2]
                    )

c7_cpd = TabularCPD("KM", 2, [[0.5,0.5,0.5,0.9865,0.5,
                                    0.1667,0.1667,0.1667,0.5,0.5,
                                    0.0156,0.1667,0.5,0.5,0.3333,
                                    0.0385,0.5,0.5,0.5,0.0294,
                                    0.5,0.5,0.5,0.5,0.4,
                                    0.0385,0.5,0.0217,0.5,0.5,
                                    0.2143,0.5,0.75,0.5,0.5,
                                    0.1,0.5,0.5,0.5,0.5],
                                   [0.5,0.5,0.5,0.0135,0.5,
                                    0.8333,0.8333,0.8333,0.5,0.5,
                                    0.9844,0.8333,0.5,0.5,0.6667,
                                    0.9615,0.5,0.5,0.5,0.9706,
                                    0.5,0.5,0.5,0.5,0.6,
                                    0.9785,0.5,0.9783,0.5,0.5,
                                    0.7857,0.5,0.25,0.5,0.5,
                                    0.9,0.5,0.5,0.5,0.5]],
                    state_names = {'KM':["positive", "negative"]},
                    evidence=['Biological Material',"Stain Appearance"],
                    evidence_card=[8,5]
                    ) 

c8_cpd = TabularCPD("KM Stain intensity", 3, [[0.86,0.03,0.65,0.325,
                                               0.01,0.94,0.02,0.97,
                                               0.04,0.57,0.01,0.97,
                                               0.03,0.64,0.09,0.82], 
                                              [0.1,0.1,0.025,0.025,
                                               0.05,0.05,0.02,0.02,
                                               0.39,0.39,0.01,0.01,
                                               0.33,0.33,0.09,0.09], 
                                              [0.03,0.86,0.325,0.65,
                                               0.94,0.01,0.97,0.02,
                                               0.57,0.04,0.97,0.01,
                                               0.64,0.03, 0.82,0.09]],
                    state_names = {'C8_StainIntesity':["strong", "weak", "none"]},
                    evidence=['Biological Material',"KM"],
                    evidence_card=[8,2]
                    )

c9_cpd = TabularCPD("Combur", 2, [[0.5,0.5,0.5,0.9865,0.5,
                                    0.25,0.1667,0.1667,0.5,0.5,
                                    0.0156,0.1667,0.5,0.5,0.3333,
                                    0.0385,0.5,0.5,0.5,0.0294,
                                    0.5,0.5,0.5,0.5,0.9,
                                    0.0385,0.5,0.0217,0.5,0.5,
                                    0.0714,0.5,0.75,0.5,0.5,
                                    0.1,0.5,0.5,0.5,0.5],
                                   [0.5,0.5,0.5,0.0417,0.5,
                                    0.75,0.8333,0.8333,0.5,0.5,
                                    0.9844,0.8333,0.5,0.5,0.5833,
                                    0.9615,0.5,0.5,0.5,0.9706,
                                    0.5,0.5,0.5,0.5,0.1,
                                    0.9615,0.5,0.9783,0.5,0.5,
                                    0.9286,0.5,0.25,0.5,0.5,
                                    0.9,0.5,0.5,0.5,0.5]],
                    state_names = {'Combur':["positive", "negative"]},
                    evidence=['Biological Material',"Stain Appearance"],
                    evidence_card=[8,5]
                    ) 

c10_cpd = TabularCPD("Combur Stain intensity", 3, [[0.95,0.04,0.07,0.8,0.04,0.925,0.02,0.97,
                                                    0.04,0.13,0.01,0.97,0.03,0.74,0.09,0.82], 
                                                   [0.004,0.004,0.13,0.13,0.04,0.04,0.02,0.02,
                                                    0.83,0.83,0.01,0.01,0.23,0.23,0.09,0.09],
                                                   [0.04,0.95,0.8,0.07,0.925,0.04,0.97,0.02,
                                                    0.13,0.04,0.97,0.01,0.74,0.03,0.82,0.09]],
                    state_names = {'C10_StainIntesity':["strong", "weak", "none"]},
                    evidence=['Biological Material',"Combur"],
                    evidence_card=[8,2]
                    )

c11_cpd = TabularCPD("Hematrace (KM)", 2, [[0.5,0.5,0.5,0.9865,0.5,0.75,0.25,0.75,0.5,0.5,
                                            0.1563,0.25,0.5,0.5,0.875,0.0714,0.5,0.5,0.5,0.0556,
                                            0.5,0.5,0.5,0.5,0.2,0.0833,0.5,0.0417,0.5,0.5,
                                            0.0714,0.5,0.8333,0.5,0.5,0.1667,0.5,0.5,0.5,0.5], 
                                           [0.5,0.5,0.5,0.0135,0.5,0.25,0.75,0.25,0.5,0.5,
                                            0.8437,0.75,0.5,0.5,0.125,0.9286,0.5,0.5,0.5,0.9444,
                                            0.5,0.5,0.5,0.5,0.8,0.9167,0.5,0.9583,0.5,0.5,
                                            0.9286,0.5,0.1667,0.5,0.5,0.8333,0.5,0.5,0.5,0.5]],
                    state_names = {'Hematrace_KM':["positive", "negative"]},
                    evidence=['Biological Material',"Stain Appearance"],
                    evidence_card=[8,5]
                    ) 

c12_cpd = TabularCPD("Hematrace (KM) Stain intensity", 3, [[0.718803],[ 0.0895792], [0.191618]],
                    state_names = {'C12_StainIntesity':["strong", "weak", "none"]},
                    evidence=['Biological Material',"Combur"],
                    evidence_card=[8,2]
                    )

c13_cpd = TabularCPD("Hematrace (Combur)", 2, [[0.632465], [0.367535]],
                    state_names = {'Hematrace_Combur':["positive", "negative"]},
                    evidence=['Biological Material',"Stain Appearance"],
                    evidence_card=[8,5]
                    ) 

c14_cpd = TabularCPD("Hematrace (Combur) Stain intensity", 3, [[0.7394], [0.0895792], [0.171021]],
                    state_names = {'C14_StainIntesity':["strong", "weak", "none"]},
                    evidence=['Biological Material',"Combur"],
                    evidence_card=[8,2]
                    )

Hp_cpd = TabularCPD("Hp", 2, [[0.5], [0.5]],
                    state_names ={'Hp':['state0', 'state1']},
                    evidence=['Hypotheses'],
                    evidence_card=[2]
                    )

Hd_cpd = TabularCPD("Hd", 2, [[0.5], [0.5]],
                    state_names = {'Hd':['state0', 'state1']},
                    evidence=['Hypotheses'],
                    evidence_card=[2]
                    )

DAG.add_cpds = (c1_cpd,c2_cpd,c3_cpd,c4_cpd,c5_cpd,c6_cpd,c7_cpd,
                 c8_cpd,c9_cpd,c10_cpd,c11_cpd,c12_cpd,c13_cpd,
                 c14_cpd,Hp_cpd,Hd_cpd)


DAG.get_cpds('Combur')

# DAG.get_independencies('Biological Material')

# DAG.check_model()
bp = BeliefPropagation(DAG)

pos = nx.drawing.nx_agraph.graphviz_layout(
        DAG,
        prog='dot',
        args='-Grankdir=TB -Eminlen=20'
    )


pos['LR'] = (pos['Hypotheses'][0]+ 200,pos['Hypotheses'][1])
pos['Hp'] = (pos['Hypotheses'][0]+ 100,pos['Hypotheses'][1]+350)
pos['Hd'] = (pos['Hypotheses'][0]+ 100,pos['Hypotheses'][1]-350)
pos['Hematrace (Combur) Stain intensity'] = (pos['Hematrace (Combur)'][0]+ 150,pos['Hematrace (Combur)'][1])
pos['Hematrace (KM) Stain intensity'] = (pos['Hematrace (KM)'][0]+ 150,pos['Hematrace (KM)'][1])
pos['KM Stain intensity'] = (pos['KM'][0],pos['KM'][1]- 1000)
pos['Combur Stain intensity'] = (pos['Combur'][0],pos['Combur'][1]- 1000)
pos['Luminol Stain intensity'] = (pos['Luminol'][0]- 150,pos['Luminol'][1])

nx.draw(DAG, 
        pos=pos, 
        node_color='#00b4d9', 
        node_size=6250,
        node_shape='D',
        with_labels=True)
plt.show()

import bnlearn as bn

query = bn.inference.fit(DAG, variables=['Combur'], evidence={'smoke':1, 'lung':1}, verbose=3)
query.df

print(query)

# graphvizDAG = DAG.to_graphviz()
# graphvizDAG.draw('BloodDAG.png', prog='neato')

# print(graphvizDAG)

from pgmpy.models import BayesianNetwork
from pgmpy.factors.discrete import TabularCPD
from pgmpy.inference import BeliefPropagation
G = BayesianNetwork([('diff', 'grade'), ('intel', 'grade'),
                   ('intel', 'SAT'), ('grade', 'letter')])
diff_cpd = TabularCPD('diff', 2, [[0.2], [0.8]])
intel_cpd = TabularCPD('intel', 3, [[0.5], [0.3], [0.2]])
grade_cpd = TabularCPD('grade', 3,
                       [[0.1, 0.1, 0.1, 0.1, 0.1, 0.1],
                        [0.1, 0.1, 0.1, 0.1, 0.1, 0.1],
                        [0.8, 0.8, 0.8, 0.8, 0.8, 0.8]],
                       evidence=['diff', 'intel'],
                       evidence_card=[2, 3])
sat_cpd = TabularCPD('SAT', 2,
                     [[0.1, 0.2, 0.7],
                      [0.9, 0.8, 0.3]],
                     evidence=['intel'], evidence_card=[3])
letter_cpd = TabularCPD('letter', 2,
                        [[0.1, 0.4, 0.8],
                         [0.9, 0.6, 0.2]],
                        evidence=['grade'], evidence_card=[3])
G.add_cpds(diff_cpd, intel_cpd, grade_cpd, sat_cpd, letter_cpd)

pos = nx.drawing.nx_agraph.graphviz_layout(
        G,
        prog='dot')
nx.draw(G, 
        pos=pos, 
        node_color='#00b4d9', 
        node_size=6250,
        # node_shape='D',
        with_labels=True)
plt.show()

bp = BeliefPropagation(G)
bp.calibrate()

G.get_cpds()
