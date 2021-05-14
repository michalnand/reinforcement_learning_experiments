import RLAgents

import matplotlib.pyplot as plt

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

'''
files = []
files.append("./models/ppo_entropy_sparse/result/result.log")
stats_entropy_sparse = RLAgents.RLStatsCompute(files) 
'''


plt.cla()
plt.ylabel("score")
plt.xlabel("iteration")
plt.grid(color='black', linestyle='-', linewidth=0.1)

plt.plot(stats_baseline.mean[0], stats_baseline.mean[8], label="dense baseline", color='gray')
plt.fill_between(stats_baseline.mean[0], stats_baseline.lower[8], stats_baseline.upper[8], color='gray', alpha=0.2)

plt.plot(stats_baseline_sparse.mean[0], stats_baseline_sparse.mean[8], label="sparse baseline", color='deepskyblue')
plt.fill_between(stats_baseline_sparse.mean[0], stats_baseline_sparse.lower[8], stats_baseline_sparse.upper[8], color='deepskyblue', alpha=0.2)

plt.plot(stats_curiosity_sparse.mean[0], stats_curiosity_sparse.mean[8], label="sparse RND", color='limegreen')
plt.fill_between(stats_curiosity_sparse.mean[0], stats_curiosity_sparse.lower[8], stats_curiosity_sparse.upper[8], color='limegreen', alpha=0.2)

#plt.plot(stats_entropy_sparse.mean[0], stats_entropy_sparse.mean[8], label="sparse RND+entropy", color='red')
#plt.fill_between(stats_entropy_sparse.mean[0], stats_entropy_sparse.lower[8], stats_entropy_sparse.upper[8], color='red', alpha=0.2)

plt.legend(loc='upper left', borderaxespad=0.)
plt.savefig(result_path + "score_per_iteration.png", dpi = 300)



plt.cla()
plt.ylabel("score")
plt.xlabel("iteration")
plt.grid(color='black', linestyle='-', linewidth=0.1)

plt.plot(stats_baseline.mean[0], stats_baseline.mean[4], label="dense baseline", color='gray')
plt.fill_between(stats_baseline.mean[0], stats_baseline.lower[4], stats_baseline.upper[4], color='gray', alpha=0.2)

plt.plot(stats_baseline_sparse.mean[0], stats_baseline_sparse.mean[4], label="sparse baseline", color='deepskyblue')
plt.fill_between(stats_baseline_sparse.mean[0], stats_baseline_sparse.lower[4], stats_baseline_sparse.upper[4], color='deepskyblue', alpha=0.2)

plt.plot(stats_curiosity_sparse.mean[0], stats_curiosity_sparse.mean[4], label="sparse RND", color='limegreen')
plt.fill_between(stats_curiosity_sparse.mean[0], stats_curiosity_sparse.lower[4], stats_curiosity_sparse.upper[4], color='limegreen', alpha=0.2)

#plt.plot(stats_entropy_sparse.mean[0], stats_entropy_sparse.mean[4], label="sparse RND+entropy", color='red')
#plt.fill_between(stats_entropy_sparse.mean[0], stats_entropy_sparse.lower[4], stats_entropy_sparse.upper[4], color='red', alpha=0.2)

plt.legend(loc='upper left', borderaxespad=0.)
plt.savefig(result_path + "score_per_iteration_normalised.png", dpi = 300)




plt.cla()
plt.ylabel("life episode length")
plt.xlabel("iteration")
plt.grid(color='black', linestyle='-', linewidth=0.1)

plt.plot(stats_baseline.mean[0], stats_baseline.mean[2], label="dense baseline", color='gray')
plt.fill_between(stats_baseline.mean[0], stats_baseline.lower[2], stats_baseline.upper[2], color='gray', alpha=0.2)

plt.plot(stats_baseline_sparse.mean[0], stats_baseline_sparse.mean[2], label="sparse baseline", color='deepskyblue')
plt.fill_between(stats_baseline_sparse.mean[0], stats_baseline_sparse.lower[2], stats_baseline_sparse.upper[2], color='deepskyblue', alpha=0.2)

plt.plot(stats_curiosity_sparse.mean[0], stats_curiosity_sparse.mean[2], label="sparse RND", color='limegreen')
plt.fill_between(stats_curiosity_sparse.mean[0], stats_curiosity_sparse.lower[2], stats_curiosity_sparse.upper[2], color='limegreen', alpha=0.2)

#plt.plot(stats_entropy_sparse.mean[0], stats_entropy_sparse.mean[2], label="sparse RND+entropy", color='red')
#plt.fill_between(stats_entropy_sparse.mean[0], stats_entropy_sparse.lower[2], stats_entropy_sparse.upper[2], color='red', alpha=0.2)

plt.legend(loc='upper left', borderaxespad=0.)
plt.savefig(result_path + "episode_length.png", dpi = 300)






plt.cla()
plt.ylabel("value")
plt.xlabel("iteration")
plt.grid(color='black', linestyle='-', linewidth=0.1)

plt.plot(stats_curiosity_sparse.mean[0], stats_curiosity_sparse.mean[10], label="curiosity", color='deepskyblue')
plt.fill_between(stats_curiosity_sparse.mean[0], stats_curiosity_sparse.lower[10], stats_curiosity_sparse.upper[10], color='deepskyblue', alpha=0.2)

plt.legend(loc='upper right', borderaxespad=0.)
plt.savefig(result_path + "sparse_rnd_internal_motivation.png", dpi = 300)


'''
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