# Tips
# Use Anaconda Prompt to locate the work directory, then open with VSCODE, the debugging should be fine

import sys, os
sys.path.append(os.path.join(sys.path[0], '..', '..'))

import Code.Model1.InhomoDiriNeumBVP as DNBVP
import numpy as np

ps = DNBVP.ProblemSetting(option=-1)
coeff = np.ones((ps.fine_grid, ps.fine_grid))
ps.set_coeff(coeff)
val, vec, op_list = ps.get_eigen_pair(5)
print(val)
