import RLAgents
import numpy

import matplotlib.pyplot as plt


raw_score_col           = 3
normalized_score_col    = 4

result_path = "./results/"


files = []
files.append("./models/ppo_curiosity/result/result.log")
stats_curiosity = RLAgents.RLStatsCompute(files) 

'''
files = []
files.append("./models/ppo_entropy/result/result.log")
stats_entropy = RLAgents.RLStatsCompute(files) 
'''


plt.cla()
plt.ylabel("score")
plt.xlabel("iteration")
plt.grid(color='black', linestyle='-', linewidth=0.1)

plt.plot(stats_curiosity.mean[0], stats_curiosity.mean[raw_score_col], label="RND", color='limegreen')
#plt.plot(stats_entropy.mean[0], stats_entropy.mean[raw_score_col], label="RND + entropy", color='red')

plt.legend(loc='upper left', borderaxespad=0.)
plt.savefig(result_path + "score_per_iteration.png", dpi = 300)


plt.cla()
plt.ylabel("score")
plt.xlabel("episode")
plt.grid(color='black', linestyle='-', linewidth=0.1)

plt.plot(stats_curiosity.mean[1], stats_curiosity.mean[normalized_score_col], label="RND", color='limegreen')
#plt.plot(stats_entropy.mean[1], stats_entropy.mean[normalized_score_col], label="RND + entropy", color='red')

plt.legend(loc='upper left', borderaxespad=0.)
plt.savefig(result_path + "score_per_iteration_normalized.png", dpi = 300)




#curiosity agent
plt.cla()
plt.ylabel("internal motivation")
plt.xlabel("iteration")
plt.grid(color='black', linestyle='-', linewidth=0.1)

plt.plot(stats_curiosity.mean[0], stats_curiosity.mean[7], label="curiosity", color='deepskyblue')

plt.legend(loc='upper right', borderaxespad=0.)

plt.savefig(result_path + "rnd_internal_motivation.png", dpi = 300)

plt.cla()
plt.ylabel("advantages")
plt.xlabel("iteration")
plt.grid(color='black', linestyle='-', linewidth=0.1)

plt.plot(stats_curiosity.mean[0], stats_curiosity.mean[8], label="external advantages", color='deepskyblue')
plt.plot(stats_curiosity.mean[0], numpy.clip(stats_curiosity.mean[9], -0.02, 0.02), label="curiosity advantages", color='limegreen')

plt.legend(loc='lower right', borderaxespad=0.)
plt.savefig(result_path + "rnd_advantages.png", dpi = 300)



'''
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
'''