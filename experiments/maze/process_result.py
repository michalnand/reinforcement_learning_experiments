import RLAgents

import matplotlib.pyplot as plt

result_path = "./results/"

files = []
files.append("./models/ppo_baseline/result/result.log")
rl_stats_compute_ppo = RLAgents.RLStatsCompute(files)

files = []
files.append("./models/ppo_curiosity/result/result.log")
rl_stats_compute_curiosity = RLAgents.RLStatsCompute(files) 

files = []
files.append("./models/ppo_entropy/result/result.log")
rl_stats_compute_entropy = RLAgents.RLStatsCompute(files) 


 

plt.cla()
plt.ylabel("score")
plt.xlabel("iteration")
plt.grid(color='black', linestyle='-', linewidth=0.1)

plt.plot(rl_stats_compute_ppo.mean[0], rl_stats_compute_ppo.mean[4], label="ppo baseline", color='deepskyblue')
plt.fill_between(rl_stats_compute_ppo.mean[0], rl_stats_compute_ppo.lower[4], rl_stats_compute_ppo.upper[4], color='deepskyblue', alpha=0.2)

plt.plot(rl_stats_compute_curiosity.mean[0], rl_stats_compute_curiosity.mean[4], label="ppo RND", color='limegreen')
plt.fill_between(rl_stats_compute_curiosity.mean[0], rl_stats_compute_curiosity.lower[4], rl_stats_compute_curiosity.upper[4], color='limegreen', alpha=0.2)

plt.plot(rl_stats_compute_entropy.mean[0], rl_stats_compute_entropy.mean[4], label="ppo RND+entropy", color='red')
plt.fill_between(rl_stats_compute_entropy.mean[0], rl_stats_compute_entropy.lower[4], rl_stats_compute_entropy.upper[4], color='red', alpha=0.2)

plt.legend(loc='lower right', borderaxespad=0.)
plt.savefig(result_path + "ppo_score_per_iteration.png", dpi = 300)


'''
plt.cla()
plt.ylabel("value")
plt.xlabel("iteration")
plt.grid(color='black', linestyle='-', linewidth=0.1)

plt.plot(rl_stats_compute_curiosity.mean[0], rl_stats_compute_curiosity.mean[10], label="curiosity", color='deepskyblue')
plt.fill_between(rl_stats_compute_curiosity.mean[0], rl_stats_compute_curiosity.lower[10], rl_stats_compute_curiosity.upper[10], color='deepskyblue', alpha=0.2)

plt.legend(loc='upper right', borderaxespad=0.)
plt.savefig(result_path + "ppo_curiosity_internal_motivation.png", dpi = 300)
'''

'''
plt.cla()
plt.ylabel("value")
plt.xlabel("iteration")
plt.grid(color='black', linestyle='-', linewidth=0.1)

plt.plot(rl_stats_compute_entropy.mean[0], rl_stats_compute_entropy.mean[10], label="curiosity", color='deepskyblue')
plt.fill_between(rl_stats_compute_entropy.mean[0], rl_stats_compute_entropy.lower[10], rl_stats_compute_entropy.upper[10], color='deepskyblue', alpha=0.2)

plt.plot(rl_stats_compute_entropy.mean[0], rl_stats_compute_entropy.mean[12], label="entropy", color='red')
plt.fill_between(rl_stats_compute_entropy.mean[0], rl_stats_compute_entropy.lower[12], rl_stats_compute_entropy.upper[12], color='red', alpha=0.2)

plt.legend(loc='upper right', borderaxespad=0.)
plt.savefig(result_path + "ppo_entropy_internal_motivation.png", dpi = 300)
'''