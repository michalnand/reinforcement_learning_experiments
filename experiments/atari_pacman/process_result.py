import RLAgents
import numpy

import matplotlib.pyplot as plt


raw_score_col           = 3
normalized_score_col    = 4

result_path = "./results/"


files = [] 
files.append("./models/ppo_baseline/result/result.log")
stats_baseline = RLAgents.RLStatsCompute(files)

files = [] 
files.append("./models/ppo_baseline_sparse/result/result.log")
stats_baseline_sparse = RLAgents.RLStatsCompute(files)

files = []
files.append("./models/ppo_curiosity_sparse/result/result.log")
stats_curiosity_sparse = RLAgents.RLStatsCompute(files) 

files = []
files.append("./models/ppo_entropy_sparse/result/result.log")
stats_entropy_sparse = RLAgents.RLStatsCompute(files) 



plt.cla()
plt.ylabel("score")
plt.xlabel("iteration")
plt.grid(color='black', linestyle='-', linewidth=0.1)

plt.plot(stats_baseline.mean[0], stats_baseline.mean[raw_score_col], label="baseline - dense rewards", color='gray')
plt.plot(stats_baseline_sparse.mean[0], stats_baseline_sparse.mean[raw_score_col], label="baseline", color='deepskyblue')
plt.plot(stats_curiosity_sparse.mean[0], stats_curiosity_sparse.mean[raw_score_col], label="RND", color='limegreen')
plt.plot(stats_entropy_sparse.mean[0], stats_entropy_sparse.mean[raw_score_col], label="RND + entropy", color='red')

plt.legend(loc='upper left', borderaxespad=0.)
plt.savefig(result_path + "score_per_iteration.png", dpi = 300)


plt.cla()
plt.ylabel("score")
plt.xlabel("iteration")
plt.grid(color='black', linestyle='-', linewidth=0.1)

plt.plot(stats_baseline_sparse.mean[0], stats_baseline_sparse.mean[normalized_score_col], label="baseline", color='deepskyblue')
plt.plot(stats_curiosity_sparse.mean[0], stats_curiosity_sparse.mean[normalized_score_col], label="RND", color='limegreen')
plt.plot(stats_entropy_sparse.mean[0], stats_entropy_sparse.mean[normalized_score_col], label="RND + entropy", color='red')

plt.legend(loc='upper left', borderaxespad=0.)
plt.savefig(result_path + "score_per_iteration_normalized.png", dpi = 300)




#curiosity agent
plt.cla()
plt.ylabel("internal motivation")
plt.xlabel("iteration")
plt.grid(color='black', linestyle='-', linewidth=0.1)

plt.plot(stats_curiosity_sparse.mean[0], stats_curiosity_sparse.mean[7], label="curiosity", color='deepskyblue')
plt.fill_between(stats_curiosity_sparse.mean[0], stats_curiosity_sparse.lower[7], stats_curiosity_sparse.upper[7], color='deepskyblue', alpha=0.2)

plt.legend(loc='upper right', borderaxespad=0.)

plt.savefig(result_path + "rnd_internal_motivation.png", dpi = 300)

plt.cla()
plt.ylabel("advantages")
plt.xlabel("iteration")
plt.grid(color='black', linestyle='-', linewidth=0.1)

plt.plot(stats_curiosity_sparse.mean[0], stats_curiosity_sparse.mean[8], label="external advantages", color='deepskyblue')
plt.plot(stats_curiosity_sparse.mean[0], 0.5*numpy.clip(stats_curiosity_sparse.mean[9], -0.1, 0.1), label="curiosity advantages", color='limegreen')

plt.legend(loc='lower right', borderaxespad=0.)
plt.savefig(result_path + "rnd_advantages.png", dpi = 300)



#curiosity + entropy agent
plt.cla() 
plt.ylabel("internal motivation")
plt.xlabel("iteration")
plt.grid(color='black', linestyle='-', linewidth=0.1)

plt.plot(stats_entropy_sparse.mean[0], stats_entropy_sparse.mean[9], label="curiosity", color='deepskyblue', alpha=0.5)
plt.plot(stats_entropy_sparse.mean[0], stats_entropy_sparse.mean[10], label="entropy", color='red', alpha=0.5)


plt.legend(loc='upper right', borderaxespad=0.)
plt.savefig(result_path + "entropy_internal_motivation.png", dpi = 300)



plt.cla() 
plt.ylabel("advantages")
plt.xlabel("iteration")
plt.grid(color='black', linestyle='-', linewidth=0.1)

plt.plot(stats_entropy_sparse.mean[0], stats_entropy_sparse.mean[11], label="external advantages", color='deepskyblue', alpha=0.5)
plt.plot(stats_entropy_sparse.mean[0], stats_entropy_sparse.mean[12], label="curiosity advantages", color='limegreen', alpha=0.5)
plt.plot(stats_entropy_sparse.mean[0], stats_entropy_sparse.mean[13], label="entropy advantages", color='red', alpha=0.5)

plt.legend(loc='upper right', borderaxespad=0.)
plt.savefig(result_path + "entropy_advantages.png", dpi = 300)
