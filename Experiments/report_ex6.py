import numpy as np
line_end = ' \\\\ \n'
table = ''
table += 'Model problem 3\n'
N_ov = 4

table = ''
table += '$H$ & $1/20$ & $1/40$ & $1/80$' +line_end
## H, numEig=3, contrast ratio=1.0e4
err_l2 =np.array([[0.014577,0.001656,0.000057,0.000057],
         [0.014700,0.005510,0.000022,0.000018],
        [0.014800,0.011192,0.000066,0.000008]]).T.tolist()
err_eg =np.array([[0.085043,0.028329,0.005065,0.004987],
        [0.087863,0.050964,0.003455,0.002532],
        [0.091359,0.072630,0.005809,0.001736]]).T.tolist()
l2_norm=np.array([[0.235810,0.015544,0.015544,0.015544],
        [0.015544,0.015544,0.015544,0.015544],
        [0.015544,0.015544,0.015544,0.015544]]).T.tolist()
eg_norm=np.array([[0.015544,0.235810,0.235810,0.235810],
        [0.235810,0.235810,0.235810,0.235810],
        [0.235810,0.235810,0.235810,0.235810]]).T.tolist()


for i in range(N_ov):
    table+= f'$E^{i}_a$'+'&'.join(['\\num{{{:.3e}}}'.format(err_eg[i][j]/eg_norm[i][j]) for j in range(3)]) +line_end
# table+= '$\|u\|_a$' +'&'.join(['\\num{{{:.3e}}}'.format(v) for v in eg_norm])+line_end
for i in range(N_ov):
    table+= f'$E^{i}_L$'+'&'.join(['\\num{{{:.3e}}}'.format(err_l2[i][j]/l2_norm[i][j]) for j in range(3)]) +line_end

table+='-'*80 +line_end
## H=1/80, numEig=3, contrast ratio=1.0e3 -> 1.0e6
err_l2=[0.015931,0.014800,0.014689,0.014678]
err_eg=[0.133180,0.091359,0.084796,0.084089]
l2_norm=[0.016873,0.015544,0.015412,0.015399]
eg_norm=[0.259732,0.235810,0.232775,0.232461]

ratios= [10**i for i in [3,4,5,6]]
table+= '$\kappa_1/\kappa_m$&'+'&'.join(['\\num{{{:.3e}}}'.format(v) for v in ratios])+line_end
table+='$E^1_a$&'+'&'.join(['\\num{{{:.3e}}}'.format(err_eg[i]/eg_norm[i]) for i in range(4)])+line_end
table+='$\|u\|_{a(\Omega)}$&'+'&'.join(['\\num{{{:.3e}}}'.format(eg_norm[i]) for i in range(4)])+line_end
table+='$E^1_L$&'+'&'.join(['\\num{{{:.3e}}}'.format(err_l2[i]/l2_norm[i]) for i in range(4)])+line_end
table+='$\|u\|_{L^2(\Omega)}$&'+'&'.join(['\\num{{{:.3e}}}'.format(l2_norm[i]) for i in range(4)])+line_end

## H=1/80, numEig=1->, contrast ratio=1.0e4, N_{ov}=3
err_l2=[0.005979,0.000070,0.000066,0.000060]
err_eg=[0.053081,0.006162,0.005809,0.005580]
l2_norm=[0.015544,0.015544,0.015544,0.015544]
eg_norm=[0.235810,0.235810,0.235810,0.235810]
table+='-'*80+line_end

numEig= [1,2,3,4]
table+= '$l_m$&'+'&'.join(['\\num{{{:.1e}}}'.format(v) for v in numEig])+line_end
table+='$E^3_a$&'+'&'.join(['\\num{{{:.3e}}}'.format(err_eg[i]/eg_norm[i]) for i in range(4)])+line_end
table+='$\|u\|_{a(\Omega)}$&'+'&'.join(['\\num{{{:.3e}}}'.format(eg_norm[i]) for i in range(4)])+line_end
table+='$E^3_L$&'+'&'.join(['\\num{{{:.3e}}}'.format(err_l2[i]/l2_norm[i]) for i in range(4)])+line_end
table+='$\|u\|_{L^2(\Omega)}$&'+'&'.join(['\\num{{{:.3e}}}'.format(l2_norm[i]) for i in range(4)])+line_end



print(table)
