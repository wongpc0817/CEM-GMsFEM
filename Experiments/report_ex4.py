import numpy as np
line_end = ' \\\\ \n'
table = ''
table += 'Model problem 2\n'
table += '-' * 80 + '\n'

#ratio = 10^4, H=1/20, l_m=3, N_ov=1->4
err_l2=np.array([[0.014793,0.001683,0.000017,0.000016],
                [0.014924,0.005584,0.000044,0.000040],
                [0.015028,0.011337,0.000088,0.000050]]).T.tolist()
err_eg=np.array([[0.086782,0.028302,0.001756,0.001509],
                [0.089634,0.051678,0.003955,0.003156],
                [0.092893,0.073729,0.006870,0.003957]]).T.tolist()
l2_norm=np.array([[0.015749,0.015749,0.015749,0.015749],
                [0.015749,0.015749,0.015749,0.015749],
                [0.015749,0.015749,0.015749,0.015749]]).T.tolist()
eg_norm=np.array([[0.220558,0.220558,0.220558,0.220558],
                [0.220558,0.220558,0.220558,0.220558],
                [0.220558,0.220558,0.220558,0.220558]]).T.tolist()

Hlist = [1/i for i in [20,40,80]]
table += '$H$&' +'&'.join(['\\num{{{:1f}}}'.format(v) for v  in Hlist])+line_end
for i in range(4):
    table+=f'$E^{i}_a$&'+'&'.join(['\\num{{{:.3e}}}'.format(err_eg[i][j]/eg_norm[i][j]) for j in range(3)])+line_end
for i in range(4):
    table+=f'$E^{i}_L$&'+'&'.join(['\\num{{{:.3e}}}'.format(err_l2[i][j]/l2_norm[i][j]) for j in range(3)])+line_end

table+='-'*80 +line_end
#H=1/80, ratio=10^3-> , l_m=3
err_l2=[0.018517,0.015028,0.0147110,0.014680]
err_eg=[0.143817,0.092893,0.0849320,0.084072]
l2_norm=[0.019419,0.015749,0.0154100,0.015377]
eg_norm=[0.250827,0.220558,0.2167690,0.216378]

ratios= [10**i for i in range(3,7)]
table+= '$\kappa_1/\kappa_m$&'+'&'.join(['\\num{{{:.3e}}}'.format(v) for v in ratios])+line_end
table+='$E^1_a$&'+'&'.join(['\\num{{{:.3e}}}'.format(err_eg[i]/eg_norm[i])for i in range(4)])+line_end
table+='$\|u\|_{a(\Omega)}$&'+'&'.join(['\\num{{{:.3e}}}'.format(eg_norm[i])for i in range(4)])+line_end
table+='$E^1_L$&'+'&'.join(['\\num{{{:.3e}}}'.format(err_l2[i]/l2_norm[i])for i in range(4)])+line_end
table+='$\|u\|_{L^2(\Omega)}$&'+'&'.join(['\\num{{{:.3e}}}'.format(l2_norm[i])for i in range(4)])+line_end

table+='-'*80+line_end
# H=1/80, ratio=10^4, N_ov=3
err_l2=[0.006061,0.000091,0.000088,0.000084]
err_eg=[0.053911,0.006984,0.006870,0.006675]
l2_norm=[0.015749,0.015749,0.015749,0.015749]
eg_norm=[0.220558,0.220558,0.220558,0.220558]

numEig =[1,2,3,4]
table+= '$l_m$&'+'&'.join([f'{v}' for v in numEig])+line_end

table+='$E^3_a$&'+'&'.join(['\\num{{{:.3e}}}'.format(err_eg[i]/eg_norm[i])for i in range(4)])+line_end
table+='$\|u\|_{a(\Omega)}$&'+'&'.join(['\\num{{{:.3e}}}'.format(eg_norm[i])for i in range(4)])+line_end
table+='$E^3_L$&'+'&'.join(['\\num{{{:.3e}}}'.format(err_l2[i]/l2_norm[i])for i in range(4)])+line_end
table+='$\|u\|_{L^2(\Omega)}$&'+'&'.join(['\\num{{{:.3e}}}'.format(l2_norm[i])for i in range(4)])+line_end


print(table)
