import sys
sys.path.insert(0,'../')

from utils.plot_utils import *

source_path = "../experiments/atari_hard/montezuma_revenge/models/"
result_path = "./atari_hard/montezuma_revenge/"

runs_count  = 3


##############################################################

note = ""
note+= "augmentations experiments \n"
note+= "intrinsic reward scaling : {0.25, 0.5, 1.0}\n"
note+= "augmentations 0 : [noise]\n"
note+= "augmentations 1 : [random_tiles, noise]\n"
note+= "augmentations 2 : [pixelate, random_tiles, noise]\n"

labels  = []
names   = []
colors  = []

labels.append("noise")
labels.append("random_tiles+noise")
labels.append("pixelate+random_tiles+noise")

names.append("ppo_cndsa_0")
names.append("ppo_cndsa_1")
names.append("ppo_cndsa_2")

colors.append("deepskyblue") 
colors.append("royalblue")
colors.append("blueviolet")


for name in names:    
    files = []

    for run in range(runs_count):
        files.append(source_path + name + "_" + str(run) + "/result/result.log")
    
    plot_cndsa(files, result_path, name)


files_runs  = []
for name in names:
    runs = []
    for run in range(runs_count):
        runs.append(source_path + name + "_" + str(run) + "/result/result.log")
    files_runs.append(runs)

plot_summary_score(files_runs, labels, colors, result_path, "cndsa_0_1_2", notes=note)


##############################################################

note = ""
note+= "intrinsic reward scaling experiments \n"
note+= "intrinsic reward scaling : {0.25, 0.5, 1.0}\n"
note+= "augmentations : [pixelate, random_tiles, noise]\n"

labels  = []
names   = []
colors  = []

labels.append("c=0.25")
labels.append("c=0.5")
labels.append("c=1.0")

names.append("ppo_cndsa_2")
names.append("ppo_cndsa_3")
names.append("ppo_cndsa_4")

colors.append("deepskyblue") 
colors.append("royalblue")
colors.append("blueviolet")


files_runs  = []
for name in names:
    runs = []
    for run in range(runs_count):
        runs.append(source_path + name + "_" + str(run) + "/result/result.log")
    files_runs.append(runs)

plot_summary_score(files_runs, labels, colors, result_path, "cndsa_2_3_4", notes=note)





##############################################################

note = ""
note+= "agent with advantages variance normalisation\n"
note+= "intrinsic reward scaling : {0.25, 0.5, 1.0}\n"
note+= "augmentations : [pixelate, random_tiles, noise]\n"

labels  = []
names   = []
colors  = []

labels.append("c=0.25")
labels.append("c=0.5") 
labels.append("c=1.0")

names.append("ppo_cndsa_5")
names.append("ppo_cndsa_6")
names.append("ppo_cndsa_7")

colors.append("deepskyblue") 
colors.append("royalblue")
colors.append("blueviolet")


for name in names:    
    files = []

    for run in range(runs_count):
        files.append(source_path + name + "_" + str(run) + "/result/result.log")
    
    plot_cndsa(files, result_path, name)



files_runs  = []
for name in names:
    runs = []
    for run in range(runs_count):
        runs.append(source_path + name + "_" + str(run) + "/result/result.log")
    files_runs.append(runs)


plot_summary_score(files_runs, labels, colors, result_path, "cndsa_5_6_7", notes=note)

##############################################################
