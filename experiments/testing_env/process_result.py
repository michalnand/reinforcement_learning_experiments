import RLAgents
import numpy

import matplotlib.pyplot as plt


raw_score_col = 3
norm_score_col = 4

result_path = "./results/"

files = [] 
files.append("./models/ppo_baseline/run_0/result/result.log")
files.append("./models/ppo_baseline/run_1/result/result.log")
files.append("./models/ppo_baseline/run_2/result/result.log")
files.append("./models/ppo_baseline/run_3/result/result.log")
files.append("./models/ppo_baseline/run_4/result/result.log")
files.append("./models/ppo_baseline/run_5/result/result.log")
files.append("./models/ppo_baseline/run_6/result/result.log")
files.append("./models/ppo_baseline/run_7/result/result.log")
stats_baseline = RLAgents.RLStatsCompute(files)

files = []
files.append("./models/ppo_curiosity/run_0/result/result.log")
files.append("./models/ppo_curiosity/run_1/result/result.log")
files.append("./models/ppo_curiosity/run_2/result/result.log")
files.append("./models/ppo_curiosity/run_3/result/result.log")
files.append("./models/ppo_curiosity/run_4/result/result.log")
files.append("./models/ppo_curiosity/run_5/result/result.log")
files.append("./models/ppo_curiosity/run_6/result/result.log")
files.append("./models/ppo_curiosity/run_7/result/result.log")
stats_curiosity = RLAgents.RLStatsCompute(files) 


files = []
files.append("./models/ppo_entropy/run_0/result/result.log")
files.append("./models/ppo_entropy/run_1/result/result.log")
files.append("./models/ppo_entropy/run_2/result/result.log")
files.append("./models/ppo_entropy/run_3/result/result.log")
files.append("./models/ppo_entropy/run_4/result/result.log")
files.append("./models/ppo_entropy/run_5/result/result.log")
files.append("./models/ppo_entropy/run_6/result/result.log")
files.append("./models/ppo_entropy/run_7/result/result.log")
stats_entropy = RLAgents.RLStatsCompute(files) 



plt.cla()
plt.ylabel("score")
plt.xlabel("iteration")
plt.grid(color='black', linestyle='-', linewidth=0.1)

plt.plot(stats_baseline.mean[0], stats_baseline.mean[raw_score_col], label="baseline", color='deepskyblue')
plt.fill_between(stats_baseline.mean[0], stats_baseline.lower[4], stats_baseline.upper[4], color='deepskyblue', alpha=0.2)

plt.plot(stats_curiosity.mean[0], stats_curiosity.mean[raw_score_col], label="RND", color='limegreen')
plt.fill_between(stats_curiosity.mean[0], stats_curiosity.lower[4], stats_curiosity.upper[4], color='limegreen', alpha=0.2)

plt.plot(stats_entropy.mean[0], stats_entropy.mean[raw_score_col], label="RND + entropy", color='red')
plt.fill_between(stats_entropy.mean[0], stats_entropy.lower[4], stats_entropy.upper[4], color='red', alpha=0.2)


plt.legend(loc='upper left', borderaxespad=0.)
plt.savefig(result_path + "score_per_iteration.png", dpi = 300)





#curiosity agent
plt.cla()
plt.ylabel("internal motivation")
plt.xlabel("iteration")
plt.grid(color='black', linestyle='-', linewidth=0.1)

plt.plot(stats_curiosity.mean[0], stats_curiosity.mean[7], label="curiosity", color='deepskyblue')
plt.fill_between(stats_curiosity.mean[0], stats_curiosity.lower[7], stats_curiosity.upper[7], color='deepskyblue', alpha=0.2)

plt.legend(loc='upper right', borderaxespad=0.)

plt.savefig(result_path + "rnd_internal_motivation.png", dpi = 300)

plt.cla()
plt.ylabel("advantages")
plt.xlabel("iteration")
plt.grid(color='black', linestyle='-', linewidth=0.1)

plt.plot(stats_curiosity.mean[0], stats_curiosity.mean[8], label="external advantages", color='deepskyblue')
plt.fill_between(stats_curiosity.mean[0], stats_curiosity.lower[8], stats_curiosity.upper[8], color='deepskyblue', alpha=0.2)

plt.plot(stats_curiosity.mean[0], stats_curiosity.mean[9], label="curiosity advantages", color='limegreen')
plt.fill_between(stats_curiosity.mean[0], stats_curiosity.lower[9], stats_curiosity.upper[9], color='limegreen', alpha=0.2)

plt.legend(loc='lower right', borderaxespad=0.)
plt.savefig(result_path + "rnd_advantages.png", dpi = 300)




#curiosity + entropy agent
plt.cla() 
plt.ylabel("internal motivation")
plt.xlabel("iteration")
plt.grid(color='black', linestyle='-', linewidth=0.1)

plt.plot(stats_entropy.mean[0], stats_entropy.mean[8], label="curiosity", color='deepskyblue', alpha=0.5)
plt.fill_between(stats_entropy.mean[0], stats_entropy.lower[8], stats_entropy.upper[8], color='deepskyblue', alpha=0.2)

plt.plot(stats_entropy.mean[0], stats_entropy.mean[9], label="entropy", color='red', alpha=0.5)
plt.fill_between(stats_entropy.mean[0], stats_entropy.lower[9], stats_entropy.upper[9], color='red', alpha=0.2)

plt.legend(loc='upper right', borderaxespad=0.)
plt.savefig(result_path + "entropy_internal_motivation.png", dpi = 300)



plt.cla() 
plt.ylabel("advantages")
plt.xlabel("iteration")
plt.grid(color='black', linestyle='-', linewidth=0.1)

plt.plot(stats_entropy.mean[0], stats_entropy.mean[10], label="external advantages", color='deepskyblue', alpha=0.5)
plt.fill_between(stats_entropy.mean[0], stats_entropy.lower[10], stats_entropy.upper[10], color='deepskyblue', alpha=0.2)

plt.plot(stats_entropy.mean[0], stats_entropy.mean[11], label="curiosity advantages", color='limegreen', alpha=0.5)
plt.fill_between(stats_entropy.mean[0], stats_entropy.lower[11], stats_entropy.upper[11], color='limegreen', alpha=0.2)

plt.plot(stats_entropy.mean[0], stats_entropy.mean[12], label="entropy advantages", color='red', alpha=0.5)
plt.fill_between(stats_entropy.mean[0], stats_entropy.lower[12], stats_entropy.upper[12], color='red', alpha=0.2)

plt.legend(loc='upper right', borderaxespad=0.)
plt.savefig(result_path + "entropy_advantages.png", dpi = 300)
