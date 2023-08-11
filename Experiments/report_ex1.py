import numpy as np
line_end = ' \\\\ \n'
table = ''
table += 'Model problem 1\n'
lambdas = [2.298640,2.299099,2.299145]
eg_norm = [68.392766,216.318225,684.071373]
l2_norm = [0.007564,0.007567,0.007567]
ratios=[10**i for i in [4,5,6]]
numEig = len(eg_norm)
table += '-' * 80 + '\n'
table += '$\kappa_1/\kappa_m$&'+'&'.join(['\\num{{{:.3e}}}'.format(v) for v in ratios])+line_end
table += '$\|u\|_a$ & ' + ' & '.join(['\\num{{{:.3e}}}'.format(v) for v in eg_norm]) + line_end
table += '$\|u\|_{L^2(\Omega)}$ & ' + ' & '.join(['\\num{{{:.3e}}}'.format(v) for v in l2_norm]) + line_end
table += '$\Lambda^\prime$ &'+'&'.join(['\\num{{{:.3e}}}'.format(lambdas[i])for i in range(numEig)])+line_end
err_eg = np.array([[0.829295,  0.023058,  0.000548,  0.000025],
                   [2.621876,  0.072883,  0.001731,  0.000078],
                   [0.000292,  0.000009,  0.000000,  0.000000]]).T.tolist()

err_l2 = np.array([[0.000292,  0.000009,  0.000000,  0.000000],
                   [0.000292,  0.000009,  0.000000,  0.000000],
                   [8.290914,  0.230465,  0.005474,  0.000245]]).T.tolist()

table += '-' * 80 + '\n'
table += '$\kappa_1/\kappa_m$&'+'&'.join(['\\num{{{:.3e}}}'.format(v) for v in ratios])+line_end
table += '$E^1_a$ & ' + ' & '.join(['\\num{{{:.3e}}}'.format(err_eg[0][i] / eg_norm[i]) for i in range(numEig)]) + line_end
table += '$E^2_a$ & ' + ' & '.join(['\\num{{{:.3e}}}'.format(err_eg[1][i] / eg_norm[i]) for i in range(numEig)]) + line_end
table += '$E^3_a$ & ' + ' & '.join(['\\num{{{:.3e}}}'.format(err_eg[2][i] / eg_norm[i]) for i in range(numEig)]) + line_end
table += '$E^4_a$ & ' + ' & '.join(['\\num{{{:.3e}}}'.format(err_eg[3][i] / eg_norm[i]) for i in range(numEig)]) + line_end

table += '-' * 80 + '\n'
table += '$\kappa_1/\kappa_m$&'+'&'.join(['\\num{{{:.3e}}}'.format(v) for v in ratios])+line_end
table += '$E^1_L$ & ' + ' & '.join(['\\num{{{:.3e}}}'.format(err_l2[0][i] / l2_norm[i]) for i in range(numEig)]) + line_end
table += '$E^2_L$ & ' + ' & '.join(['\\num{{{:.3e}}}'.format(err_l2[1][i] / l2_norm[i]) for i in range(numEig)]) + line_end
table += '$E^3_L$ & ' + ' & '.join(['\\num{{{:.3e}}}'.format(err_l2[2][i] / l2_norm[i]) for i in range(numEig)]) + line_end
table += '$E^4_L$ & ' + ' & '.join(['\\num{{{:.3e}}}'.format(err_l2[3][i] / l2_norm[i]) for i in range(numEig)]) + line_end

print(table)
