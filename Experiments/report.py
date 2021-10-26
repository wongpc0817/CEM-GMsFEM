line_end = ' \\\\ \n'
table = ''
table += 'Model problem 1\n'
ref_eg_norm = [2.825539, 2.841413, 2.843136, 2.843314]
ref_l2_norm = [1.853289, 1.852970, 1.852954, 1.852952]
table += '-' * 80 + '\n'
table += 'Contrast & \\num{{1.e+3}} & \\num{{1.e+4}} & \\num{{1.e+5}} & \\num{{1.e+6}}' + line_end
table += 'Ref_EG & ' + ' & '.join(['\\num{{{:.3e}}}'.format(v) for v in ref_eg_norm]) + line_end
table += 'Ref_L2 & ' + ' & '.join(['\\num{{{:.3e}}}'.format(v) for v in ref_l2_norm]) + line_end

abs_err_eg = [[5.494217, 17.129617, 54.075910, 170.972912], [0.505778, 1.250468, 3.016379, 8.534713], [0.022271, 0.065394, 0.201764, 0.590112], [0.001114, 0.002940, 0.008929, 0.028097]]
abs_err_l2 = [[0.145787, 0.149382, 0.149833, 0.149877], [0.024654, 0.084888, 0.122884, 0.134340], [0.000055, 0.000429, 0.004029, 0.030700], [0.000002, 0.000002, 0.000017, 0.000023]]
table += '-' * 80 + '\n'
table += 'Layers\\Contrast & \\num{{1.e+3}} & \\num{{1.e+4}} & \\num{{1.e+5}} & \\num{{1.e+6}}' + line_end
table += '$1$ & ' + ' & '.join(['\\num{{{:.3e}}}'.format(abs_err_eg[0][i] / ref_eg_norm[i]) for i in range(4)]) + line_end
table += '$2$ & ' + ' & '.join(['\\num{{{:.3e}}}'.format(abs_err_eg[1][i] / ref_eg_norm[i]) for i in range(4)]) + line_end
table += '$3$ & ' + ' & '.join(['\\num{{{:.3e}}}'.format(abs_err_eg[2][i] / ref_eg_norm[i]) for i in range(4)]) + line_end
table += '$4$ & ' + ' & '.join(['\\num{{{:.3e}}}'.format(abs_err_eg[3][i] / ref_eg_norm[i]) for i in range(4)]) + line_end

table += '-' * 80 + '\n'
table += 'Layers\\Contrast & \\num{{1.e+3}} & \\num{{1.e+4}} & \\num{{1.e+5}} & \\num{{1.e+6}}' + line_end
table += '$1$ & ' + ' & '.join(['\\num{{{:.3e}}}'.format(abs_err_l2[0][i] / ref_l2_norm[i]) for i in range(4)]) + line_end
table += '$2$ & ' + ' & '.join(['\\num{{{:.3e}}}'.format(abs_err_l2[1][i] / ref_l2_norm[i]) for i in range(4)]) + line_end
table += '$3$ & ' + ' & '.join(['\\num{{{:.3e}}}'.format(abs_err_l2[2][i] / ref_l2_norm[i]) for i in range(4)]) + line_end
table += '$4$ & ' + ' & '.join(['\\num{{{:.3e}}}'.format(abs_err_l2[3][i] / ref_l2_norm[i]) for i in range(4)]) + line_end

abs_err_eg = [[2.188372, 4.128269, 8.709003, 17.129633], [0.114306, 0.231887, 0.569639, 1.250464], [0.007564, 0.007479, 0.022030, 0.065393], [0.006557, 0.001217, 0.000864, 0.002940]]
abs_err_l2 = [[0.128902, 0.122355, 0.137954, 0.149384], [0.001258, 0.005998, 0.030835, 0.084886], [0.000131, 0.000013, 0.000053, 0.000429], [0.000123, 0.000009, 0.000002, 0.000002]]
table += '-' * 80 + '\n'
table += 'Layers\\H & $1/10$ & $1/20$ & $1/40$ & $1/80$ ' + line_end
table += '$1$ & ' + ' & '.join(['\\num{{{:.3e}}}'.format(abs_err_eg[0][i] / ref_eg_norm[1]) for i in range(4)]) + line_end
table += '$2$ & ' + ' & '.join(['\\num{{{:.3e}}}'.format(abs_err_eg[1][i] / ref_eg_norm[1]) for i in range(4)]) + line_end
table += '$3$ & ' + ' & '.join(['\\num{{{:.3e}}}'.format(abs_err_eg[2][i] / ref_eg_norm[1]) for i in range(4)]) + line_end
table += '$4$ & ' + ' & '.join(['\\num{{{:.3e}}}'.format(abs_err_eg[3][i] / ref_eg_norm[1]) for i in range(4)]) + line_end

table += '-' * 80 + '\n'
table += 'Layers\\H & $1/10$ & $1/20$ & $1/40$ & $1/80$ ' + line_end
table += '$1$ & ' + ' & '.join(['\\num{{{:.3e}}}'.format(abs_err_l2[0][i] / ref_l2_norm[1]) for i in range(4)]) + line_end
table += '$2$ & ' + ' & '.join(['\\num{{{:.3e}}}'.format(abs_err_l2[1][i] / ref_l2_norm[1]) for i in range(4)]) + line_end
table += '$3$ & ' + ' & '.join(['\\num{{{:.3e}}}'.format(abs_err_l2[2][i] / ref_l2_norm[1]) for i in range(4)]) + line_end
table += '$4$ & ' + ' & '.join(['\\num{{{:.3e}}}'.format(abs_err_l2[3][i] / ref_l2_norm[1]) for i in range(4)]) + line_end

abs_err_eg = [2.273714, 1.401320, 0.065394, 0.059918]
abs_err_l2 = [0.116688, 0.066503, 0.000429, 0.000371]
table += '-' * 80 + '\n'
table += 'Eigen_num & $1$ & $2$ & $3$ & $4$ ' + line_end
table += 'Err_EG & ' + ' & '.join(['\\num{{{:.3e}}}'.format(abs_err_eg[i] / ref_eg_norm[1]) for i in range(4)]) + line_end
table += 'Err_L2 & ' + ' & '.join(['\\num{{{:.3e}}}'.format(abs_err_l2[i] / ref_l2_norm[1]) for i in range(4)]) + line_end

corr_eg_norm = [67.671799, 214.045789, 676.887672]
corr_l2_norm = [0.006947, 0.006952, 0.006953]
abs_err_eg = [[0.711969, 2.249941, 7.114462], [0.017426, 0.054970, 0.173787], [0.000452, 0.001411, 0.004456], [0.000020, 0.000060, 0.000190]]
abs_err_l2 = [[0.000274, 0.000274, 0.000274], [0.000011, 0.000011, 0.000011], [0.000001, 0.000001, 0.000001], [0.000000, 0.000000, 0.000000]]
max_lambda = [1.148600, 1.148829, 1.148852]
table += '-' * 80 + '\n'
table += 'Layers\\Contrast & \\num{{1.e+4}} & \\num{{1.e+5}} & \\num{{1.e+6}} ' + line_end
table += '$1$ & ' + ' & '.join(['\\num{{{:.3e}}}'.format(abs_err_eg[0][i] / corr_eg_norm[i]) for i in range(3)]) + line_end
table += '$2$ & ' + ' & '.join(['\\num{{{:.3e}}}'.format(abs_err_eg[1][i] / corr_eg_norm[i]) for i in range(3)]) + line_end
table += '$3$ & ' + ' & '.join(['\\num{{{:.3e}}}'.format(abs_err_eg[2][i] / corr_eg_norm[i]) for i in range(3)]) + line_end
table += '$4$ & ' + ' & '.join(['\\num{{{:.3e}}}'.format(abs_err_eg[3][i] / corr_eg_norm[i]) for i in range(3)]) + line_end
table += 'corr_eg_norm &' + ' & '.join(['\\num{{{:.3e}}}'.format(corr_eg_norm[i]) for i in range(3)]) + line_end
table += 'Lambda & ' + ' & '.join(['\\num{{{:.3e}}}'.format(max_lambda[i]) for i in range(3)]) + line_end

table += '-' * 80 + '\n'
table += 'Layers\\Contrast & \\num{{1.e+4}} & \\num{{1.e+5}} & \\num{{1.e+6}} ' + line_end
table += '$1$ & ' + ' & '.join(['\\num{{{:.3e}}}'.format(abs_err_l2[0][i] / corr_l2_norm[i]) for i in range(3)]) + line_end
table += '$2$ & ' + ' & '.join(['\\num{{{:.3e}}}'.format(abs_err_l2[1][i] / corr_l2_norm[i]) for i in range(3)]) + line_end
table += '$3$ & ' + ' & '.join(['\\num{{{:.3e}}}'.format(abs_err_l2[2][i] / corr_l2_norm[i]) for i in range(3)]) + line_end
table += '$4$ & ' + ' & '.join(['\\num{{{:.3e}}}'.format(abs_err_l2[3][i] / corr_l2_norm[i]) for i in range(3)]) + line_end
table += 'corr_l2_norm & ' + ' & '.join(['\\num{{{:.3e}}}'.format(corr_l2_norm[i]) for i in range(3)]) + line_end

table += 'Model problem 2\n'
corr_eg_norm = [0.186970, 0.197865, 0.198811, 0.198905, 0.198914, 0.198915]
corr_l2_norm = [0.002113, 0.002338, 0.002360, 0.002362, 0.002362, 0.002362]
abs_err_eg = [[0.001837, 0.001967, 0.001978, 0.001979, 0.001979, 0.001979], [0.000124, 0.000062, 0.000038, 0.000035, 0.000034, 0.000034], [0.000014, 0.000008, 0.000003, 0.000001, 0.000001, 0.000001]]
abs_err_l2 = [[0.000012, 0.000019, 0.000020, 0.000020, 0.000020, 0.000020], [0.000001, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000], [0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000]]
max_lambda = [0.797787, 0.872628, 0.880241, 0.881004, 0.881080, 0.881087]
table += 'Layers\\Contrast & \\num{1.e+1} & \\num{1.e+2} & \\num{1.e+3} & \\num{1.e+4} & \\num{1.e+5} & \\num{1.e+6} ' + line_end
table += '$\Lambda$ & ' + ' & '.join(['\\num{{{:.3e}}}'.format(max_lambda[i]) for i in range(6)]) + line_end
table += 'EG & ' + ' & '.join(['\\num{{{:.3e}}}'.format(corr_eg_norm[i]) for i in range(6)]) + line_end
table += '$1$ & ' + ' & '.join(['\\num{{{:.3e}}}'.format(abs_err_eg[0][i] / corr_eg_norm[i]) for i in range(6)]) + line_end
table += '$2$ & ' + ' & '.join(['\\num{{{:.3e}}}'.format(abs_err_eg[1][i] / corr_eg_norm[i]) for i in range(6)]) + line_end
table += '$3$ & ' + ' & '.join(['\\num{{{:.3e}}}'.format(abs_err_eg[2][i] / corr_eg_norm[i]) for i in range(6)]) + line_end
table += 'L2 & ' + ' & '.join(['\\num{{{:.3e}}}'.format(corr_l2_norm[i]) for i in range(6)]) + line_end
table += '$1$ & ' + ' & '.join(['\\num{{{:.3e}}}'.format(abs_err_l2[0][i] / corr_l2_norm[i]) for i in range(6)]) + line_end
table += '$2$ & ' + ' & '.join(['\\num{{{:.3e}}}'.format(abs_err_l2[1][i] / corr_l2_norm[i]) for i in range(6)]) + line_end
table += '-' * 80 + '\n'

abs_err_eg = [[0.146653, 0.094024, 0.085760, 0.084867], [0.063582, 0.077165, 0.082028, 0.082666], [0.003235, 0.006958, 0.019640, 0.050215], [0.000188, 0.000312, 0.000790, 0.002416]]
abs_err_l2 = [[0.018607, 0.015079, 0.014759, 0.014728], [0.006950, 0.012433, 0.014300, 0.014548], [0.000019, 0.000102, 0.000820, 0.005368], [0.000000, 0.000000, 0.000001, 0.000012]]
ref_eg_norm = [0.250827, 0.220558, 0.216769, 0.216378]
ref_l2_norm = [0.019419, 0.015749, 0.015410, 0.015377]
table += '-' * 80 + '\n'
table += 'Layers\\Contrast & \\num{1.e+3} & \\num{1.e+4} & \\num{1.e+5} & \\num{1.e+6}' + line_end
table += '$1$ & ' + ' & '.join(['\\num{{{:.3e}}}'.format(abs_err_eg[0][i] / ref_eg_norm[i]) for i in range(4)]) + line_end
table += '$2$ & ' + ' & '.join(['\\num{{{:.3e}}}'.format(abs_err_eg[1][i] / ref_eg_norm[i]) for i in range(4)]) + line_end
table += '$3$ & ' + ' & '.join(['\\num{{{:.3e}}}'.format(abs_err_eg[2][i] / ref_eg_norm[i]) for i in range(4)]) + line_end
table += '$4$ & ' + ' & '.join(['\\num{{{:.3e}}}'.format(abs_err_eg[3][i] / ref_eg_norm[i]) for i in range(4)]) + line_end
table += 'EG & ' + ' & '.join(['\\num{{{:.3e}}}'.format(ref_eg_norm[i]) for i in range(4)]) + line_end

table += '-' * 80 + '\n'
table += 'Layers\\Contrast & \\num{1.e+3} & \\num{1.e+4} & \\num{1.e+5} & \\num{1.e+6}' + line_end
table += '$1$ & ' + ' & '.join(['\\num{{{:.3e}}}'.format(abs_err_l2[0][i] / ref_l2_norm[i]) for i in range(4)]) + line_end
table += '$2$ & ' + ' & '.join(['\\num{{{:.3e}}}'.format(abs_err_l2[1][i] / ref_l2_norm[i]) for i in range(4)]) + line_end
table += '$3$ & ' + ' & '.join(['\\num{{{:.3e}}}'.format(abs_err_l2[2][i] / ref_l2_norm[i]) for i in range(4)]) + line_end
table += '$4$ & ' + ' & '.join(['\\num{{{:.3e}}}'.format(abs_err_l2[3][i] / ref_l2_norm[i]) for i in range(4)]) + line_end
table += 'L2 & ' + ' & '.join(['\\num{{{:.3e}}}'.format(ref_l2_norm[i]) for i in range(4)]) + line_end
table += '-' * 80 + '\n'
table += 'Model problem 3\n'
abs_err_eg = [[0.136766, 0.092754, 0.085779, 0.085026], [0.055171, 0.076077, 0.081914, 0.082655], [0.002820, 0.006869, 0.019615, 0.050209], [0.000172, 0.000308, 0.000789, 0.002415]]
abs_err_l2 = [[0.016016, 0.014850, 0.014737, 0.014726], [0.006094, 0.012276, 0.014282, 0.014546], [0.000017, 0.000101, 0.000819, 0.005368], [0.000000, 0.000000, 0.000001, 0.000012]]
ref_eg_norm = [0.258383, 0.234220, 0.231130, 0.230810]
ref_l2_norm = [0.016869, 0.015541, 0.015408, 0.015395]
table += '-' * 80 + '\n'
table += 'Layers\\Contrast & \\num{1.e+3} & \\num{1.e+4} & \\num{1.e+5} & \\num{1.e+6}' + line_end
table += '$1$ & ' + ' & '.join(['\\num{{{:.3e}}}'.format(abs_err_eg[0][i] / ref_eg_norm[i]) for i in range(4)]) + line_end
table += '$2$ & ' + ' & '.join(['\\num{{{:.3e}}}'.format(abs_err_eg[1][i] / ref_eg_norm[i]) for i in range(4)]) + line_end
table += '$3$ & ' + ' & '.join(['\\num{{{:.3e}}}'.format(abs_err_eg[2][i] / ref_eg_norm[i]) for i in range(4)]) + line_end
table += '$4$ & ' + ' & '.join(['\\num{{{:.3e}}}'.format(abs_err_eg[3][i] / ref_eg_norm[i]) for i in range(4)]) + line_end
table += 'EG & ' + ' & '.join(['\\num{{{:.3e}}}'.format(ref_eg_norm[i]) for i in range(4)]) + line_end

table += '-' * 80 + '\n'
table += 'Layers\\Contrast & \\num{1.e+3} & \\num{1.e+4} & \\num{1.e+5} & \\num{1.e+6}' + line_end
table += '$1$ & ' + ' & '.join(['\\num{{{:.3e}}}'.format(abs_err_l2[0][i] / ref_l2_norm[i]) for i in range(4)]) + line_end
table += '$2$ & ' + ' & '.join(['\\num{{{:.3e}}}'.format(abs_err_l2[1][i] / ref_l2_norm[i]) for i in range(4)]) + line_end
table += '$3$ & ' + ' & '.join(['\\num{{{:.3e}}}'.format(abs_err_l2[2][i] / ref_l2_norm[i]) for i in range(4)]) + line_end
table += '$4$ & ' + ' & '.join(['\\num{{{:.3e}}}'.format(abs_err_l2[3][i] / ref_l2_norm[i]) for i in range(4)]) + line_end
table += 'L2 & ' + ' & '.join(['\\num{{{:.3e}}}'.format(ref_l2_norm[i]) for i in range(4)]) + line_end

print(table)
