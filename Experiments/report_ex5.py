import numpy as np
line_end = ' \\\\ \n'
table = ''
table += 'Model problem 3\n'
lambdas = [1.763144,1.763296,1.763312]
eg_norm = [0.214521,0.214530,0.214531]
l2_norm = [0.002494,0.002494,0.002494]
numEig = len(eg_norm)
table += '-' * 80 + '\n'
table += '$\kappa_1/\kappa_m$ & \\num{1.000e+3} & \\num{1.000e+4} & \\num{1.000e+5} & \\num{1.000e+6}' + line_end
table += '$\|u\|_a$ & ' + ' & '.join(['\\num{{{:.3e}}}'.format(v) for v in eg_norm]) + line_end
table += '$\|u\|_{L^2(\Omega)}$ & ' + ' & '.join(['\\num{{{:.3e}}}'.format(v) for v in l2_norm]) + line_end
table += '$\Lambda^\prime$ &'+'&'.join(['\\num{{{:.3e}}}'.format(lambdas[i])for i in range(numEig)])+line_end
err_eg = np.array([[0.001949,  0.000032,  0.000001,  0.000000],
                   [0.001949,  0.000032,  0.000001,  0.000000],
                   [0.001949,  0.000032,  0.000001,  0.000000]]).T.tolist()

err_l2 = np.array([[0.000019,  0.000000,  0.000000,  0.000000],
                   [0.000019,  0.000000,  0.000000,  0.000000],
                   [0.000019,  0.000000,  0.000000,  0.000000]]).T.tolist()


table += '-' * 80 + '\n'
table += 'Layers\\Contrast & \\num{1.000e+3} & \\num{1.000e+4} & \\num{1.000e+5} & \\num{1.000e+6}' + line_end
table += '$E^1_a$ & ' + ' & '.join(['\\num{{{:.3e}}}'.format(err_eg[0][i] / eg_norm[i]) for i in range(numEig)]) + line_end
table += '$E^2_a$ & ' + ' & '.join(['\\num{{{:.3e}}}'.format(err_eg[1][i] / eg_norm[i]) for i in range(numEig)]) + line_end
table += '$E^3_a$ & ' + ' & '.join(['\\num{{{:.3e}}}'.format(err_eg[2][i] / eg_norm[i]) for i in range(numEig)]) + line_end
table += '$E^4_a$ & ' + ' & '.join(['\\num{{{:.3e}}}'.format(err_eg[3][i] / eg_norm[i]) for i in range(numEig)]) + line_end

table += '-' * 80 + '\n'
table += 'Layers\\Contrast & \\num{1.000e+3} & \\num{1.000e+4} & \\num{1.000e+5} & \\num{1.000e+6}' + line_end
table += '$E^1_L$ & ' + ' & '.join(['\\num{{{:.3e}}}'.format(err_l2[0][i] / l2_norm[i]) for i in range(numEig)]) + line_end
table += '$E^2_L$ & ' + ' & '.join(['\\num{{{:.3e}}}'.format(err_l2[1][i] / l2_norm[i]) for i in range(numEig)]) + line_end
table += '$E^3_L$ & ' + ' & '.join(['\\num{{{:.3e}}}'.format(err_l2[2][i] / l2_norm[i]) for i in range(numEig)]) + line_end
table += '$E^4_L$ & ' + ' & '.join(['\\num{{{:.3e}}}'.format(err_l2[3][i] / l2_norm[i]) for i in range(numEig)]) + line_end

print(table)
