import sympy as sp

c_AQ = sp.symbols('c_AQ')
c_AP = sp.symbols('c_AP')
c_PQ = sp.symbols('c_PQ')
c_QA = sp.symbols('c_QA')
Myo = sp.symbols('Myo')
c_Myo = sp.symbols('c_Myo')
c_PT = sp.symbols('c_PT')
c_PSC = sp.symbols('c_PSC')
c_TM = sp.symbols('c_TM')
Cyte = sp.symbols('Cyte')
QSC = sp.symbols('QSC')
PSC = sp.symbols('PSC')
F_QSC = sp.symbols('F_QSC')
F_prime_QSC = sp.symbols('F_prime_QSC')
ASC = sp.symbols('ASC')

Jacobian = sp.Matrix([
    [-c_AQ-c_AP*F_QSC, -c_AP*ASC*F_prime_QSC+c_QA*(1-Myo/c_Myo), 0, 0, -c_QA*QSC/c_Myo],
    [c_AQ, -c_QA*(1-Myo/c_Myo), c_PQ, 0, c_QA*QSC/c_Myo],
    [c_AP*F_QSC, c_AP*ASC*F_prime_QSC, -c_PT-c_PQ+c_PSC*(1-Myo/c_Myo), 0, -c_PSC*PSC/c_Myo],
    [0, 0, c_PT, -c_TM*(1-Myo/c_Myo), c_TM*Cyte/c_Myo],
    [0, 0, 0, c_TM*(1-Myo/c_Myo), -c_TM*Cyte/c_Myo],
])

subs_dict = {
    c_AQ: 9,
    c_AP: 18,
    c_PQ: 3,
    c_QA: 20,
    c_Myo: 15000,
    c_PT: 15,
    c_PSC: 5,
    c_TM: 13,
    F_QSC: (1-286/1000),
    QSC: 286,
    ASC: 0,
    PSC: 0,
    F_prime_QSC: -1/1000,
    Myo: 15000,
    Cyte: 813,
}

J_num = Jacobian.subs(subs_dict)
eigs = J_num.evalf().eigenvals()
print(eigs)
