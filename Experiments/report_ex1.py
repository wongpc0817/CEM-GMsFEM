import numpy as np
line_end = ' \\\\ \n'
table = ''
table += 'Model problem 1\n'
lambdas = [3.319007,3.319007,3.319007]
eg_norm = [66.667711,211.044773,670.470387]
l2_norm = [0.007169,0.007174,0.007227]
ratios=[10**i for i in [4,5,6]]
numEig = len(eg_norm)
table += '-' * 80 + '\n'
table += '$\kappa_1/\kappa_m$&'+'&'.join(['\\num{{{:.3e}}}'.format(v) for v in ratios])+line_end
table += '$\|u\|_a$ & ' + ' & '.join(['\\num{{{:.3e}}}'.format(v) for v in eg_norm]) + line_end
table += '$\|u\|_{L^2(\Omega)}$ & ' + ' & '.join(['\\num{{{:.3e}}}'.format(v) for v in l2_norm]) + line_end
table += '$\Lambda^\prime$ &'+'&'.join(['\\num{{{:.3e}}}'.format(lambdas[i])for i in range(numEig)])+line_end
err_eg = np.array([[0.588712,  0.013972,  0.000488,  0.000020],
                   [1.862620,  0.044102,  0.001544,  0.000064],
                   [5.838741,  0.140616,  0.004883,  0.000204]]).T.tolist()

err_l2 = np.array([[0.000188,  0.000005,  0.000000,  0.000000],
                   [0.000189,  0.000005,  0.000000,  0.000000],
                   [0.000186,  0.000005,  0.000000,  0.000000]]).T.tolist()

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
