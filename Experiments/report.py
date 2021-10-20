line_end = ' \\\\ \n'
table = ''
table += 'Model problem 1\n'
ref_eg_norm = [2.825539, 2.841413, 2.843136, 2.843314]
ref_l2_norm = [1.853289, 1.852970, 1.852954, 1.852952]
table += '-' * 80 + '\n'
table += 'Contrast & 1.e+3 & 1.e+4 & 1.e+5 & 1.e+6' + line_end
table += 'Ref_EG & ' + ' & '.join(['{:.3f}'.format(v) for v in ref_eg_norm]) + line_end
table += 'Ref_L2 & ' + ' & '.join(['{:.3f}'.format(v) for v in ref_l2_norm]) + line_end

abs_err_eg = [[5.494217, 17.129617, 54.075910, 170.972912], [0.505778, 1.250468, 3.016379, 8.534713], [0.022271, 0.065394, 0.201764, 0.590112], [0.001114, 0.002940, 0.008929, 0.028097]]
abs_err_l2 = [[0.145787, 0.149382, 0.149833, 0.149877], [0.024654, 0.084888, 0.122884, 0.134340], [0.000055, 0.000429, 0.004029, 0.030700], [0.000002, 0.000002, 0.000017, 0.000023]]
table += '-' * 80 + '\n'
table += 'Layers\\Contrast & 1.e+3 & 1.e+4 & 1.e+5 & 1.e+6' + line_end
table += '1 & ' + ' & '.join(['{:.3f}'.format(abs_err_eg[0][i] / ref_eg_norm[i]) for i in range(4)]) + line_end
table += '2 & ' + ' & '.join(['{:.3f}'.format(abs_err_eg[1][i] / ref_eg_norm[i]) for i in range(4)]) + line_end
table += '3 & ' + ' & '.join(['{:.3f}'.format(abs_err_eg[2][i] / ref_eg_norm[i]) for i in range(4)]) + line_end
table += '4 & ' + ' & '.join(['{:.3f}'.format(abs_err_eg[3][i] / ref_eg_norm[i]) for i in range(4)]) + line_end

table += '-' * 80 + '\n'
table += 'Layers\\Contrast & 1.e+3 & 1.e+4 & 1.e+5 & 1.e+6' + line_end
table += '1 & ' + ' & '.join(['{:.3f}'.format(abs_err_l2[0][i] / ref_l2_norm[i]) for i in range(4)]) + line_end
table += '2 & ' + ' & '.join(['{:.3f}'.format(abs_err_l2[1][i] / ref_l2_norm[i]) for i in range(4)]) + line_end
table += '3 & ' + ' & '.join(['{:.3f}'.format(abs_err_l2[2][i] / ref_l2_norm[i]) for i in range(4)]) + line_end
table += '4 & ' + ' & '.join(['{:.3f}'.format(abs_err_l2[3][i] / ref_l2_norm[i]) for i in range(4)]) + line_end

abs_err_eg = [[4.128269, 8.709003, 17.129633], [0.231887, 0.569639, 1.250464], [0.007479, 0.022030, 0.065393], [0.001217, 0.000864, 0.002940]]
abs_err_l2 = [[0.122355, 0.137954, 0.149384], [0.005998, 0.030835, 0.084886], [0.000013, 0.000053, 0.000429], [0.000009, 0.000002, 0.000002]]
table += '-' * 80 + '\n'
table += 'Layers\\H & 1/20 & 1/40 & 1/80 ' + line_end
table += '1 & ' + ' & '.join(['{:.3f}'.format(abs_err_eg[0][i] / ref_eg_norm[1]) for i in range(3)]) + line_end
table += '2 & ' + ' & '.join(['{:.3f}'.format(abs_err_eg[1][i] / ref_eg_norm[1]) for i in range(3)]) + line_end
table += '3 & ' + ' & '.join(['{:.3f}'.format(abs_err_eg[2][i] / ref_eg_norm[1]) for i in range(3)]) + line_end
table += '4 & ' + ' & '.join(['{:.3f}'.format(abs_err_eg[3][i] / ref_eg_norm[1]) for i in range(3)]) + line_end

table += '-' * 80 + '\n'
table += 'Layers\\H & 1/20 & 1/40 & 1/80 ' + line_end
table += '1 & ' + ' & '.join(['{:.3f}'.format(abs_err_l2[0][i] / ref_l2_norm[1]) for i in range(3)]) + line_end
table += '2 & ' + ' & '.join(['{:.3f}'.format(abs_err_l2[1][i] / ref_l2_norm[1]) for i in range(3)]) + line_end
table += '3 & ' + ' & '.join(['{:.3f}'.format(abs_err_l2[2][i] / ref_l2_norm[1]) for i in range(3)]) + line_end
table += '4 & ' + ' & '.join(['{:.3f}'.format(abs_err_l2[3][i] / ref_l2_norm[1]) for i in range(3)]) + line_end

abs_err_eg = [2.273714, 1.401320, 0.065394, 0.059918]
abs_err_l2 = [0.116688, 0.066503, 0.000429, 0.000371]
table += '-' * 80 + '\n'
table += 'Eigen_num & 1 & 2 & 3 & 4 ' + line_end
table += 'Err_EG & ' + ' & '.join(['{:.3f}'.format(abs_err_eg[i] / ref_eg_norm[1]) for i in range(4)]) + line_end
table += 'Err_L2 & ' + ' & '.join(['{:.3f}'.format(abs_err_l2[i] / ref_l2_norm[1]) for i in range(4)]) + line_end

print(table)
