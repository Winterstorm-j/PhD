from numpy import matrix
from pgmpy.factors.discrete import TabularCPD
from pgmpy.models import BayesianNetwork
from pgmpy.inference import BeliefPropagation
import networkx as nx
import matplotlib.pyplot as plt
import pygraphviz
import matplotlib.patches as mpatches

# initiate empty DAG
DAG = BayesianNetwork()

# add nodes
DAG.add_nodes_from(['Hypotheses','Blood present','Biological Material',
                    'Stain Appearance','Luminol','Luminol Stain intensity','KM','KM Stain intensity','Combur',
                    'Combur Stain intensity','Hematrace (KM)',
                    'Hematrace (KM)\n Stain intensity','Hematrace (Combur)',
                    'Hematrace (Combur)\n Stain intensity','RSID-Blood (KM)',
                    'RSID-Blood (KM)\n Stain intensity','Hp','Hd', 'LR'])

# add edges
DAG.add_edges_from([('Hypotheses', 'Blood present'), ('Hypotheses', 'Hp'), ('Hypotheses', 'Hd'), ('Hp', 'LR'), ('Hd', 'LR'), 
    ('Blood present', 'Biological Material'), ('Biological Material', 'KM'), ('Biological Material', 'Hematrace (KM)'), ('KM', 'Hematrace (KM)'),
    ('Hematrace (KM)', 'Hematrace (KM)\n Stain intensity'), ('Biological Material', 'Hematrace (KM)\n Stain intensity'), ('Biological Material', 'KM Stain intensity'), ('KM', 'KM Stain intensity'),
    ('Biological Material', 'Stain Appearance'), ('KM', 'Stain Appearance'), ('Biological Material', 'Combur'), ('Stain Appearance', 'Combur'),
    ('Biological Material', 'Hematrace (Combur)'), ('Combur', 'Hematrace (Combur)'), ('Biological Material', 'Hematrace (Combur)\n Stain intensity'), ('Hematrace (Combur)', 'Hematrace (Combur)\n Stain intensity'),
    ('Biological Material', 'Combur Stain intensity'), ('Combur', 'Combur Stain intensity'), ('Stain Appearance', 'Luminol'), ('Biological Material', 'Luminol'),
    ('Biological Material', 'Luminol Stain intensity'), ('Luminol', 'Luminol Stain intensity'),
    ('Biological Material', 'RSID-Blood (KM)'), ('KM', 'RSID-Blood (KM)'),
    ('RSID-Blood (KM)', 'RSID-Blood (KM)\n Stain intensity'),])

# define CPT values
c1_cpd = TabularCPD('Hypotheses', 2, [[0.5], [0.5]],
                    state_names = {'Hypotheses':['Hp', 'Hd']}
                    )

c2_cpd = TabularCPD("Blood present", 2, [[1,0],
                                        [0,1]],
                    state_names = {'Blood present':["yes", "no"],
                                   'Hypotheses':['Hp', 'Hd']},
                       evidence=['Hypotheses'],
                       evidence_card=[2]
                    )

c3_cpd = TabularCPD("Biological Material", 8, 
                    [[1,0],
                        [0,0.142857],
                        [0,0.142857],
                        [0,0.142857],
                        [0,0.142857],
                        [0,0.142857],
                        [0,0.142857],
                        [0,0.142857]],
                    state_names = {'Biological Material':["Blood", "Semen", "Saliva", "Sweat", "Faeces", "Urine", "Vaginal material", "Breast milk"],
                                   'Blood present':["yes", "no"]},
                       evidence=['Blood present'],
                       evidence_card=[2]
                    )

c4_cpd = TabularCPD("Stain Appearance", 4, [
        [0.016949,0.564103,0.82703,0.645963,0.008,0.48555,0.489362,0.12037],
        [0.00565,0.222222,0.07027,0.018633,0.008,0.00578,0.177305,0.10185],
        [0.011299,0.196581,0.02703,0.008,0.49711,0.269504,0.27,0.64815],
        [0.926554,0.008547,0.0054,0.006211,0.28,0.00578,0.056737,0.00926],
        ],
                    state_names = {'Stain Appearance':["none", "white/crystaline", "white/yellow", "Red/Brown"],
                                   'Biological Material':["Blood", "Semen", "Saliva", "Sweat", "Faeces", "Urine", "Vaginal material", "Breast milk"]},
                       evidence=['Biological Material'],
                       evidence_card=[8]
                    )

c5_cpd = TabularCPD("Luminol", 2, 
                    [[0.5,0.5,0.5,0.986486,0.5,
                        0.5,0.833333,0.166667,0.5,0.5,
                        0.734375,0.833333,0.5,0.5,0.928571,
                        0.038462,0.5,0.5,0.5,0.3125,
                        0.5,0.5,0.5,0.5,0.15,
                        0.041667,0.5,0.113636,0.5,0.5,
                        0.142857,0.5,0.1,0.5,0.5,
                        0.125,0.5,0.5,0.5,0.5],
                        [0.5,0.5,0.5,0.013514,0.5,
                        0.5,0.166667,0.833333,0.5,0.5,
                        0.265625,0.166667,0.5,0.5,0.071429,
                        0.961538,0.5,0.5,0.5,0.6875,
                        0.5,0.5,0.5,0.5,0.85,
                        0.958333,0.5,0.886364,0.5,0.5,
                        0.857143,0.5,0.9,0.5,0.5,
                        0.875,0.5,0.5,0.5,0.5]],
                    state_names = {'Luminol':["positive", "negative"],
                                   'Stain Appearance':["none", "white/crystaline", "white/yellow", "Red/Brown"],
                                   'Biological Material':["Blood", "Semen", "Saliva", "Sweat", "Faeces", "Urine", "Vaginal material", "Breast milk"]},
                    evidence=['Biological Material','Stain Appearance'],
                    evidence_card=[8,5]
                    )

c6_cpd = TabularCPD("Luminol Stain intensity", 3, 
                    [[0.984375,0.0078125,0.00781255,0.8,
                        0.590476,0.4,0.186441,0.79661,
                        0.173913,0.782609,0.0140845,0.887324,
                        0.128205,0.846154,0.090909,0.727273], 
                        [0.0078125,0.0078125,0.025,0.025,
                        0.009524,0.009524,0.016949,0.016949,
                        0.043478,0.043478,0.043478,0.043478,
                        0.025641,0.025641,0.181818,0.181818],
                        [0.0078125,0.984375,0.8,0.175,
                        0.4,0.590476,0.79661,0.186441,
                        0.782609,0.173913,0.887324,0.0140845,
                        0.846154,0.128205,0.727273,0.090909]],
                    state_names = {'Luminol Stain intensity':["strong", "weak", "none"],
                                   'Luminol':["positive", "negative"],
                                   'Biological Material':["Blood", "Semen", "Saliva", "Sweat", "Faeces", "Urine", "Vaginal material", "Breast milk"]},
                    evidence=['Biological Material',"Luminol"],
                    evidence_card=[8,2]
                    )

c7_cpd = TabularCPD("KM", 2, [[0.5,0.5,0.5,0.986486,0.5,
                                0.166667,0.166667,0.166667,0.5,0.5,
                                0.015152,0.166667,0.5,0.5,0.333333,
                                0.038462,0.5,0.5,0.5,0.029412,
                                0.5,0.5,0.5,0.5,0.4,
                                0.038462,0.5,0.021739,0.5,0.5,
                                0.21429,0.5,0.75,0.5,0.5,
                                0.1,0.5,0.5,0.5,0.5],
                                [0.5,0.5,0.5,0.013514,0.5,
                                0.833333,0.833333,0.833333,0.5,0.5,
                                0.984848,0.8333,0.5,0.5,0.666667,
                                0.961538,0.5,0.5,0.5,0.970588,
                                0.5,0.5,0.5,0.5,0.6,
                                0.961538,0.5,0.978261,0.5,0.5,
                                0.78571,0.5,0.25,0.5,0.5,
                                0.9,0.5,0.5,0.5,0.5]],
                    state_names = {'KM':["positive", "negative"],
                                   'Stain Appearance':["none", "white/crystaline", "white/yellow", "Red/Brown", "other"],
                                   'Biological Material':["Blood", "Semen", "Saliva", "Sweat", "Faeces", "Urine", "Vaginal material", "Breast milk"]},
                    evidence=['Biological Material',"Stain Appearance"],
                    evidence_card=[8,5]
                    ) 

c8_cpd = TabularCPD("KM Stain intensity", 3, 
                    [[0.862069,0.0344828,0.65,0.325,
                        0.0092597,0.944444,0.016949,0.966102,
                        0.0434783,0.565217,0.0140845,0.971831,
                        0.025641,0.641026,00.090909,0.818182], 
                        [0.103448,0.103448,0.025,0.025,
                        0.0462963,0.0462963,0.016949,0.016949,
                        0.391304,0.391304,0.0140845,0.0140845,
                        0.333333,0.333333,0.090909,0.090909], 
                        [0.034483,0.862069,0.325,0.65,
                        0.944444,0.0092597,0.966102,0.966102,
                        0.565217,0.0434783,0.971831,0.0140845,
                        0.641026,0.025641, 0.818182,0.090909]],
                    state_names = {'KM Stain intensity':["strong", "weak", "none"],
                                   'KM':["positive", "negative"],
                                   'Biological Material':["Blood", "Semen", "Saliva", "Sweat", "Faeces", "Urine", "Vaginal material", "Breast milk"]},
                    evidence=['Biological Material',"KM"],
                    evidence_card=[8,2]
                    )

c9_cpd = TabularCPD("Combur", 2, 
                    [[0.5,0.5,0.5,0.958333,0.5,
                        0.25,0.166667,0.166667,0.5,0.5,
                        0.015625,0.166667,0.5,0.5,0.416667,
                        0.038462,0.5,0.5,0.5,0.029412,
                        0.5,0.5,0.5,0.5,0.9,
                        0.038462,0.5,0.021739,0.5,0.5,
                        0.071429,0.5,0.75,0.5,0.5,
                        0.1,0.5,0.5,0.5,0.5],
                        [0.5,0.5,0.5,0.041667,0.5,
                        0.75,0.833333,0.833333,0.5,0.5,
                        0.984375,0.833333,0.5,0.5,0.583333,
                        0.961538,0.5,0.5,0.5,0.970588,
                        0.5,0.5,0.5,0.5,0.1,
                        0.961538,0.5,0.978261,0.5,0.5,
                        0.928571,0.5,0.25,0.5,0.5,
                        0.9,0.5,0.5,0.5,0.5]],
                    state_names = {'Combur':["positive", "negative"],
                                   'Stain Appearance':["none", "white/crystaline", "white/yellow", "Red/Brown", "other"],
                                   'Biological Material':["Blood", "Semen", "Saliva", "Sweat", "Faeces", "Urine", "Vaginal material", "Breast milk"]},
                    evidence=['Biological Material',"Stain Appearance"],
                    evidence_card=[8,5]
                    ) 

c10_cpd = TabularCPD("Combur Stain intensity", 3, 
                     [[0.952569,0.043478,0.066667,0.8,
                        0.0375,0.925,0.016949,0.996102,
                        0.043478,0.130435,0.0140845,0.971831,
                        0.025641,0.74359,0.090909,0.818182], 
                        [0.003953,0.003953,0.133333,0.133333,
                        0.0375,0.0375,0.016949,0.016949,
                        0.826087,0.826087,0.0140845,0.0140845,
                        0.230769,0.230769,0.090909,0.090909],
                        [0.043478,0.952569,0.8,0.066667,
                        0.925,0.0375,0.966102,0.016949,
                        0.130435,0.043478,0.971831,0.0140845,
                        0.74359,0.025641,0.818182,0.090909]],
                    state_names = {'Combur Stain intensity':["strong", "weak", "none"],
                                   'Combur':["positive", "negative"],
                                   'Biological Material':["Blood", "Semen", "Saliva", "Sweat", "Faeces", "Urine", "Vaginal material", "Breast milk"]},
                    evidence=['Biological Material',"Combur"],
                    evidence_card=[8,2]
                    )

c11_cpd = TabularCPD("Hematrace (KM)", 2, 
                     [[0.964286,0.5,0.75,0.5,0.166667,0.5,0.666667,0.5,
                        0.5,0.666667,0.236842,0.033333,0.033333,0.029412,0.083333,0.166667], 
                        [0.035714,0.5,0.25,0.5,0.833333,0.5,0.333333,0.5,
                        0.5,0.333333,0.763158,0.966667,0.666667,0.970588,0.916667,0.833333]],
                    state_names = {'Hematrace (KM)':["positive", "negative"],
                                   'KM':["positive", "negative"],
                                   'Biological Material':["Blood", "Semen", "Saliva", "Sweat", "Faeces", "Urine", "Vaginal material", "Breast milk"]},
                    evidence=['Biological Material','KM'],
                    evidence_card=[8,2]
                    ) 

c12_cpd = TabularCPD("Hematrace (KM)\n Stain intensity", 3, 
                     [[0.894736,0.052632,0.4,0.4,
                       0.220588,0.544118,0.032258,0.935484,
                       0.333333,0.555556,0.02439,0.92683,
                       0.230769,0.653846,0.142857,0.714286],
                      [0.052632,0.052632,0.2,0.2,
                       0.235294,0.235294,0.032258,0.032258,
                       0.111111,0.111111,0.04878,0.04878,
                       0.115385,0.115385,0.142857,0.142857],
                       [0.052632,0.894736,0.4,0.4,
                       0.544118,0.220588,0.935484,0.032258,
                       0.555556,0.333333,0.92683,0.02439,
                       0.653846,0.230769,0.714286,0.142857]],
                    state_names = {'Hematrace (KM)\n Stain intensity':["strong", "weak", "none"],
                                   'Hematrace (KM)':["positive", "negative"],
                                   'Biological Material':["Blood", "Semen", "Saliva", "Sweat", "Faeces", "Urine", "Vaginal material", "Breast milk"]},
                    evidence=['Biological Material','Hematrace (KM)'],
                    evidence_card=[8,2]
                    )

c13_cpd = TabularCPD("RSID-Blood (KM)", 2, 
                    [[0.964286,0.5,0.166667,0.5,
                        0.166667,0.5,0.125,0.5,
                        0.5,0.125,0.026316,0.033333,
                        0.125,0.027778,0.071429,0.166667], 
                        [0.035714,0.5,0.833333,0.5,
                        0.833333,0.5,0.875,0.5,
                        0.5,0.875,0.973684,0.966667,
                        0.875,0.972222,0.928571,0.833333]],
                    state_names = {'RSID-Blood (KM)':["positive", "negative"],
                                   'KM':["positive", "negative"],
                                   'Biological Material':["Blood", "Semen", "Saliva", "Sweat", "Faeces", "Urine", "Vaginal material", "Breast milk"]},
                    evidence=['Biological Material',"KM"],
                    evidence_card=[8,2]
                    )

c14_cpd = TabularCPD("RSID-Blood (KM)\n Stain intensity", 3, 
                     [[0.945946, 0.027027,0.1,0.8,
                       0.016393,0.836066,0.032258,0.935484,
                       0.076923,0.846154,0.026316,0.947368,
                       0.047619,0.904762,0.125,0.75],
                      [0.027027, 0.027027,0.1,0.1,
                       0.147541,0.147541,0.032258,0.032258,
                       0.076923,0.076923,0.026316,0.026316,
                       0.047619,0.047619,0.125,0.125],
                       [0.027027,0.945946,0.8,0.1,
                       0.836066,0.016393,0.935484,0.032258,
                       0.846154,0.076923,0.947368,0.026316,
                       0.904762,0.047619,0.75,0.125]],
                    state_names = {'RSID-Blood (KM)\n Stain intensity':["strong", "weak", "none"],
                                   'RSID-Blood (KM)':["positive", "negative"],
                                   'Biological Material':["Blood", "Semen", "Saliva", "Sweat", "Faeces", "Urine", "Vaginal material", "Breast milk"]},
                    evidence=['Biological Material','RSID-Blood (KM)'],
                    evidence_card=[8,2]
                    )

c15_cpd = TabularCPD("Hematrace (Combur)", 2, 
                     [[0.995868,0.5,0.8,0.5,
                       0.222222,0.5,0.833333,0.5,
                      0.916667,0.666667,0.216216,0.033333,
                      0.333333,0.029412,0.071429,0.166667],
                      [0.004132,0.5,0.2,0.5,
                       0.777778,0.5,0.166667,0.5,
                       0.083333,0.333333,0.783784,0.966667,
                       0.666667,0.970588,0.928571,0.833333]],
                    state_names = {'Hematrace (Combur)':["positive", "negative"],
                                   'Combur':["positive", "negative"],
                                   'Biological Material':["Blood", "Semen", "Saliva", "Sweat", "Faeces", "Urine", "Vaginal material", "Breast milk"]},
                    evidence=['Biological Material','Combur'],
                    evidence_card=[8,2]
                    ) 

c16_cpd = TabularCPD("Hematrace (Combur)\n Stain intensity", 3, 
                     [[0.833333,0.052632,0.4,0.4,
                       0.220588,0.544118,0.032258,0.935484,
                       0.333333,0.555556,0.02439,0.92683,
                       0.230769,0.653846,0.142857,0.714286],
                      [0.052632,0.052632,0.2,0.2,
                       0.235294,0.235294,0.032258,0.032258,
                       0.111111,0.111111,0.04878,0.04878,
                       0.115385,0.115385,0.142857,0.142857],
                      [0.052632,0.894736,0.4,0.4,
                       0.544118,0.220588,0.935484,0.032258,
                       0.555556,0.333333,0.92683,0.02439,
                       0.653846,0.230769,0.714286,0.142857]],
                    state_names = {'Hematrace (Combur)\n Stain intensity':["strong", "weak", "none"],
                                   'Hematrace (Combur)':["positive", "negative"],
                                   'Biological Material':["Blood", "Semen", "Saliva", "Sweat", "Faeces", "Urine", "Vaginal material", "Breast milk"]},
                    evidence=['Biological Material','Hematrace (Combur)'],
                    evidence_card=[8,2]
                    )

Hp_cpd = TabularCPD("Hp", 2, [[1,0], [0,1]],
                    state_names ={'Hp':['state0', 'state1']},
                    evidence=['Hypotheses'],
                    evidence_card=[2]
                    )

Hd_cpd = TabularCPD("Hd", 2, [[1,0], [0,1]],
                    state_names = {'Hd':['state0', 'state1']},
                    evidence=['Hypotheses'],
                    evidence_card=[2]
                    )

# add CPTs to DAG
DAG.add_cpds(c1_cpd,c2_cpd,c3_cpd,c4_cpd,c5_cpd,c6_cpd,c7_cpd,
                 c8_cpd,c9_cpd,c10_cpd,c11_cpd,c12_cpd,c13_cpd,
                 c14_cpd,c15_cpd,
                 c16_cpd,Hp_cpd,Hd_cpd)

# return CPT tables for all nodes
print(DAG.get_cpds())

# DAG.get_independencies('Biological Material')

# check DAG is valid
DAG.check_model()
bp = BeliefPropagation(DAG)

bp.calibrate()

#create custom positions
pos = nx.drawing.nx_agraph.graphviz_layout(
        DAG,
        prog='dot',
        args='-Grankdir=TB -Eminlen=20'
    )

# manual positioning (pos[0] is x-axis, pos[1] is y-axis)
pos['LR'] = (pos['Hypotheses'][0]+ 200,pos['Hypotheses'][1])
pos['Hp'] = (pos['Hypotheses'][0]+ 100,pos['Hypotheses'][1]+350)
pos['Hd'] = (pos['Hypotheses'][0]+ 100,pos['Hypotheses'][1]-350)
pos['Stain Appearance'] = (pos['Biological Material'][0]- 250,pos['Biological Material'][1])
pos['Hematrace (Combur)\n Stain intensity'] = (pos['Hematrace (Combur)'][0]+ 150,pos['Hematrace (Combur)'][1])
pos['Hematrace (KM)\n Stain intensity'] = (pos['Hematrace (KM)'][0]+ 150,pos['Hematrace (KM)'][1])
pos['KM Stain intensity'] = (pos['KM'][0],pos['KM'][1]- 1000)
pos['Combur Stain intensity'] = (pos['Combur'][0],pos['Combur'][1]- 1000)
pos['Luminol Stain intensity'] = (pos['Luminol'][0]- 150,pos['Luminol'][1])

# plot DAG with nx
nx.draw(DAG, 
        pos=pos, 
        node_color='#00b4d9', 
        node_size=1000,
        node_shape='D',
        with_labels=True,
        font_size=10)
plt.show()

# infer from BN (WIP)
import bnlearn as bn

query = bn.inference.fit(DAG, variables=['Combur'], evidence={'smoke':1, 'lung':1}, verbose=3)

print(query)

# plot with graphviz
# graphvizDAG = DAG.to_graphviz()
# graphvizDAG.draw('BloodDAG.png', prog='neato')

# print(graphvizDAG)

