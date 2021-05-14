import RLAgents
import numpy

import matplotlib.pyplot as plt


raw_score_col = 3
norm_score_col = 4

result_path = "./results/"

files = []
files.append("./models/ppo_baseline/result/result.log")
stats_baseline = RLAgents.RLStatsCompute(files)

files = []
files.append("./models/ppo_curiosity/result/result.log")
stats_curiosity = RLAgents.RLStatsCompute(files) 

'''
files = []
files.append("./models/ppo_entropy_sparse/result/result.log")
stats_entropy_sparse = RLAgents.RLStatsCompute(files) 
'''


plt.cla()
plt.ylabel("score")
plt.xlabel("iteration")
plt.grid(color='black', linestyle='-', linewidth=0.1)

plt.plot(stats_baseline.mean[0], stats_baseline.mean[raw_score_col], label="baseline", color='deepskyblue')
plt.plot(stats_curiosity.mean[0], stats_curiosity.mean[raw_score_col], label="RND", color='limegreen')
#plt.plot(stats_entropy.mean[0], stats_entropy.mean[raw_score_col], label="RND+entropy", color='red')

plt.legend(loc='upper left', borderaxespad=0.)
plt.savefig(result_path + "score_per_iteration.png", dpi = 300)



plt.cla()
plt.ylabel("score normalised")
plt.xlabel("iteration")
plt.grid(color='black', linestyle='-', linewidth=0.1)

plt.plot(stats_baseline.mean[0], stats_baseline.mean[norm_score_col], label="baseline", color='deepskyblue')
plt.plot(stats_curiosity.mean[0], stats_curiosity.mean[norm_score_col], label="RND", color='limegreen')
#plt.plot(stats_entropy.mean[0], stats_entropy.mean[norm_score_col], label="RND+entropy", color='red')

plt.legend(loc='upper left', borderaxespad=0.)
plt.savefig(result_path + "score_per_iteration_normalised.png", dpi = 300)


plt.cla()
plt.ylabel("internal motivation")
plt.xlabel("iteration")
plt.grid(color='black', linestyle='-', linewidth=0.1)

plt.plot(stats_curiosity.mean[0], stats_curiosity.mean[7], label="curiosity", color='deepskyblue')
plt.legend(loc='upper right', borderaxespad=0.)

plt.savefig(result_path + "rnd_curiosity.png", dpi = 300)


plt.cla()
plt.ylabel("probability")
plt.xlabel("value")
plt.grid(color='black', linestyle='-', linewidth=0.1)

plt.plot(stats_curiosity.hist[7][0], stats_curiosity.hist[7][1], label="curiosity", color='deepskyblue')
plt.legend(loc='upper right', borderaxespad=0.)

plt.savefig(result_path + "rnd_curiosity_histogram.png", dpi = 300)


plt.cla()
plt.ylabel("advantages")
plt.xlabel("iteration")
plt.grid(color='black', linestyle='-', linewidth=0.1)

plt.plot(stats_curiosity.mean[0], stats_curiosity.mean[8], label="external advantages", color='deepskyblue')
plt.plot(stats_curiosity.mean[0], numpy.clip(stats_curiosity.mean[9], -1, 1), label="internal advantages", color='limegreen')

plt.legend(loc='lower right', borderaxespad=0.)
plt.savefig(result_path + "rnd_advantages.png", dpi = 300)


plt.cla()
plt.ylabel("probability")
plt.xlabel("value")
plt.grid(color='black', linestyle='-', linewidth=0.1)

plt.plot(stats_curiosity.hist[8][0], stats_curiosity.hist[8][1], label="external advantages", color='deepskyblue')
plt.plot(stats_curiosity.hist[9][0], stats_curiosity.hist[9][1], label="internal advantages", color='limegreen')
plt.legend(loc='upper right', borderaxespad=0.)

plt.savefig(result_path + "rnd_advantages_histogram.png", dpi = 300)



'''

plt.cla()
plt.ylabel("value")
plt.xlabel("iteration")
plt.grid(color='black', linestyle='-', linewidth=0.1)

plt.plot(stats_curiosity.mean[0], stats_curiosity.mean[10], label="curiosity", color='deepskyblue')
plt.fill_between(stats_curiosity.mean[0], stats_curiosity.lower[10], stats_curiosity.upper[10], color='deepskyblue', alpha=0.2)

plt.legend(loc='upper right', borderaxespad=0.)
plt.savefig(result_path + "sparse_rnd_internal_motivation.png", dpi = 300)


plt.cla() 
plt.ylabel("value")
plt.xlabel("iteration")
plt.grid(color='black', linestyle='-', linewidth=0.1)

plt.plot(stats_entropy_sparse.mean[0], stats_entropy_sparse.mean[10], label="curiosity", color='deepskyblue', alpha=0.5)
plt.fill_between(stats_entropy_sparse.mean[0], stats_entropy_sparse.lower[10], stats_entropy_sparse.upper[10], color='deepskyblue', alpha=0.2)

plt.plot(stats_entropy_sparse.mean[0], stats_entropy_sparse.mean[12], label="entropy", color='red', alpha=0.5)
plt.fill_between(stats_entropy_sparse.mean[0], stats_entropy_sparse.lower[12], stats_entropy_sparse.upper[12], color='red', alpha=0.2)

plt.legend(loc='upper right', borderaxespad=0.)
plt.savefig(result_path + "sparse_rnd_entropy_internal_motivation.png", dpi = 300)
'''