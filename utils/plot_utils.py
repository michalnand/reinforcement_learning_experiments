import RLAgents
import numpy
from collections import namedtuple


import matplotlib.pyplot as plt
#import networkx as nx
from matplotlib.ticker import MaxNLocator


def _add_plot(axs, labels, colors, stats, idx, from_extended = False):
    for i in range(len(stats)):

        x = stats[i].mean[0]*128/1000000
        if from_extended:
            #x = stats[i].extended_mean[0]*128/1000000
            y = stats[i].extended_mean[idx]
            y_min = stats[i].extended_lower[idx]
            y_max = stats[i].extended_upper[idx]
        else:
            #x = stats[i].mean[0]*128/1000000
            y = stats[i].mean[idx]
            y_min = stats[i].lower[idx]
            y_max = stats[i].upper[idx]

        if numpy.all((y == 0)):
            jitter = 0.06*i
        else:
            jitter = 0  

        axs.plot(x, y + jitter, label=labels[i], linewidth=2.0, color=colors[i], alpha=1.0)
        axs.fill_between(x, y_min, y_max, facecolor=colors[i], alpha=0.5)
        axs.legend()


def _get_markdown_desc(files_runs, labels, notes, output_path, output_prefix):

    f_name = output_path + output_prefix + "_notes.md"
    f = open(f_name, 'w')

    f.write("# results for " + output_prefix + "\n\n\n")
    if notes is not None: 
        notes_ = notes.replace("\n", "\n\n")
        f.write("## notes\n")
        f.write(notes_)
        f.write("\n\n\n")

    f.write("# input files and configs\n\n\n")
    for i in range(len(labels)):
        f.write("**" + labels[i] + "**" + "\n\n")
        for run in range(len(files_runs[i])):
            f.write("* run " + str(run) + "\n\n")

            f_result_name       = files_runs[i][run]
            f_result_name_full  = "../../"+f_result_name

            f_conf_name         = f_result_name.replace("result/result.log", "src/config.py")
            f_conf_name_full    = "../../"+f_conf_name


            f.write("* result: " + "[" + f_result_name +"](" + f_result_name_full + ")" + "\n\n")
            f.write("* config: " + "[" + f_conf_name +"](" + f_conf_name_full + ")" + "\n\n")

        f.write("\n")

    f.write("\n\n\n")

    f.write("# results " + "\n\n\n")
    f.write("## result in fig : " + output_prefix + ".png\n")
    f.write("![img:results](" + output_prefix + ".png)\n")
    f.write("\n\n\n")

    f.close()

def plot_summary_score(files_runs, labels, colors, output_path, output_prefix, extended_names = ["explored_rooms"], raw_score_only = False, notes = None):
    _get_markdown_desc(files_runs, labels, notes, output_path, output_prefix)


    stats = []

    for files in files_runs:
        print("processing stats for ", files)
        stat = RLAgents.RLStatsCompute(files, extended_names = extended_names)
        stats.append(stat)

    plt.clf()

    axis_count = 1

    if raw_score_only == False:
        axis_count+= 1
    if len(extended_names) > 0:
        axis_count+= 1

 
    if raw_score_only == True:
        fig, axs = plt.subplots(axis_count, 1, figsize=(10, 5))

        _add_plot(axs, labels, colors, stats, 3)

        axs.legend(loc="upper left")
        axs.set_xlabel("samples [milions]", fontweight='bold')
        axs.set_ylabel("score", fontweight='bold')
        axs.xaxis.set_major_locator(MaxNLocator(integer=True))
        axs.grid(True)
    else:
        fig, axs = plt.subplots(axis_count, 1, figsize=(10, 10))

        _add_plot(axs[0], labels, colors, stats, 3)

        axs[0].legend(loc="upper left")
        axs[0].set_xlabel("samples [milions]", fontweight='bold')
        axs[0].set_ylabel("score", fontweight='bold')
        axs[0].xaxis.set_major_locator(MaxNLocator(integer=True))
        axs[0].grid(True)


        _add_plot(axs[1], labels, colors, stats, 4)

        axs[1].legend(loc="upper left")
        axs[1].set_xlabel("samples [milions]", fontweight='bold')
        axs[1].set_ylabel("external reward", fontweight='bold')
        axs[1].xaxis.set_major_locator(MaxNLocator(integer=True))
        axs[1].grid(True)

        if len(extended_names) > 0:
            _add_plot(axs[2], labels, colors, stats, 0, True)

            axs[2].legend(loc="upper left")
            axs[2].set_xlabel("samples [milions]", fontweight='bold')
            axs[2].set_ylabel("explored rooms", fontweight='bold')
            axs[2].xaxis.set_major_locator(MaxNLocator(integer=True))
            axs[2].grid(True)
        

    fig.tight_layout()
    fig.savefig(output_path + output_prefix + ".png", dpi = 300, bbox_inches='tight', pad_inches=0.1)





'''
score           : offset 3
external reward : offset 4

from offset 6 :

6 : internal_motivation_mean 
7 : internal_motivation_std 

8 : loss_ppo_actor           
9 : loss_ppo_critic          
10 : loss_distillation        
11 : loss_target_regularization
12 : loss_target_aux          

13 : target_magnitude         
14 : target_magnitude_std     
15 : target_similarity_accuracy
16 : target_action_accuracy   
'''
def plot_cndsa(files, output_path, output_prefix, paralel_envs = 128, extended_names = ["explored_rooms"]):
    
    im_clip_range = 0.05

    stats   = RLAgents.RLStatsCompute(files, extended_names = extended_names)

    samples = stats.mean[0]*paralel_envs/1000000

    #plot scores only (single fig)

    plt.clf()

    fig, axs = plt.subplots(1, 1, figsize=(10, 4))

    axs.plot(samples, stats.mean[4], color='deepskyblue')
    axs.fill_between(samples, stats.lower[4], stats.upper[4], facecolor='deepskyblue', alpha=0.5)
    axs.set_xlabel("samples [milions]", fontweight='bold')
    axs.set_ylabel("external reward", fontweight='bold')
    axs.set_xlim(left=0)
    
    axs.xaxis.set_major_locator(MaxNLocator(integer=True))
    axs.yaxis.set_major_locator(MaxNLocator(integer=True))
    axs.grid(True)

    fig.tight_layout()
    plt.savefig(output_path + output_prefix + "_external_reward.png", dpi = 300)


    plt.clf()

    fig, axs = plt.subplots(1, 1, figsize=(10, 4))
 
    axs.plot(samples, stats.mean[3], color='deepskyblue')
    axs.fill_between(samples, stats.lower[3], stats.upper[3], facecolor='deepskyblue', alpha=0.5)
    axs.set_xlabel("samples [milions]", fontweight='bold')
    axs.set_ylabel("loss actor", fontweight='bold')
    axs.set_xlim(left=0)
    
    axs.xaxis.set_major_locator(MaxNLocator(integer=True))
    axs.yaxis.set_major_locator(MaxNLocator(integer=True))
    axs.grid(True)

    fig.tight_layout()
    plt.savefig(output_path + output_prefix + "_score.png", dpi = 300)


    #plot losses

    plt.clf()

    if stats.mean[12].mean() != 0:
        aux_loss = True
    else:
        aux_loss = False

    if aux_loss:
        fig, axs = plt.subplots(5, 1, figsize=(10, 12))
    else:
        fig, axs = plt.subplots(4, 1, figsize=(10, 10))



    axs[0].plot(samples, stats.mean[8], color='red')
    axs[0].fill_between(samples, stats.lower[8], stats.upper[4], facecolor='red', alpha=0.5)
    axs[0].set_xlabel("samples [milions]", fontweight='bold')
    axs[0].set_ylabel("loss actor", fontweight='bold')
    axs[0].set_xlim(left=0)
    
    axs[0].xaxis.set_major_locator(MaxNLocator(integer=True))
    axs[0].yaxis.set_major_locator(MaxNLocator(integer=True))
    axs[0].grid(True)


    axs[1].plot(samples, stats.mean[9], color='deepskyblue')
    axs[1].fill_between(samples, stats.lower[9], stats.upper[4], facecolor='deepskyblue', alpha=0.5)
    axs[1].set_xlabel("samples [milions]", fontweight='bold')
    axs[1].set_ylabel("loss critic", fontweight='bold')
    axs[1].set_xlim(left=0)
    
    axs[1].xaxis.set_major_locator(MaxNLocator(integer=True))
    axs[1].yaxis.set_major_locator(MaxNLocator(integer=True))
    axs[1].grid(True)


    axs[2].plot(samples, stats.mean[10], color='green')
    axs[2].fill_between(samples, stats.lower[10], stats.upper[4], facecolor='green', alpha=0.5)
    axs[2].set_xlabel("samples [milions]", fontweight='bold')
    axs[2].set_ylabel("loss distillation", fontweight='bold')
    axs[2].set_xlim(left=0)
    
    axs[2].xaxis.set_major_locator(MaxNLocator(integer=True))
    axs[2].yaxis.set_major_locator(MaxNLocator(integer=True))
    axs[2].grid(True)


    axs[3].plot(samples, stats.mean[11], color='orange')
    axs[3].fill_between(samples, stats.lower[11], stats.upper[4], facecolor='orange', alpha=0.5)
    axs[3].set_xlabel("samples [milions]", fontweight='bold')
    axs[3].set_ylabel("loss target regularization", fontweight='bold')
    axs[3].set_xlim(left=0)
    
    axs[3].xaxis.set_major_locator(MaxNLocator(integer=True))
    axs[3].yaxis.set_major_locator(MaxNLocator(integer=True))
    axs[3].grid(True)


    if aux_loss:
        axs[4].plot(samples, stats.mean[12], color='purple')
        axs[4].fill_between(samples, stats.lower[12], stats.upper[4], facecolor='purple', alpha=0.5)
        axs[4].set_xlabel("samples [milions]", fontweight='bold')
        axs[4].set_ylabel("loss aux", fontweight='bold')
        axs[4].set_xlim(left=0)
        
        axs[4].xaxis.set_major_locator(MaxNLocator(integer=True))
        axs[4].yaxis.set_major_locator(MaxNLocator(integer=True))
        axs[4].grid(True)

    fig.tight_layout()
    plt.savefig(output_path + output_prefix + "_loss.png", dpi = 300)


    #plot intrinsic motivation

    plt.clf()

   
    fig, axs = plt.subplots(4, 1, figsize=(10, 10))

    sm = numpy.clip(stats.mean[6], -im_clip_range, im_clip_range)
    sl = numpy.clip(stats.lower[6], -im_clip_range, im_clip_range)
    su = numpy.clip(stats.upper[6], -im_clip_range, im_clip_range)


    axs[0].plot(samples, sm, color='deepskyblue')
    axs[0].fill_between(samples, sl, su, facecolor='deepskyblue', alpha=0.5)
    axs[0].set_xlabel("samples [milions]", fontweight='bold')
    axs[0].set_ylabel("intrinsic motivation", fontweight='bold')
    axs[0].set_xlim(left=0)
    
    axs[0].xaxis.set_major_locator(MaxNLocator(integer=True))
    axs[0].yaxis.set_major_locator(MaxNLocator(integer=True))
    axs[0].grid(True)


    axs[1].plot(samples, stats.mean[7], color='deepskyblue')
    axs[1].fill_between(samples, stats.lower[7], stats.upper[7], facecolor='deepskyblue', alpha=0.5)
    axs[1].set_xlabel("samples [milions]", fontweight='bold')
    axs[1].set_ylabel("intrinsic motivation variance", fontweight='bold')
    axs[1].set_xlim(left=0)
    
    axs[1].xaxis.set_major_locator(MaxNLocator(integer=True))
    axs[1].yaxis.set_major_locator(MaxNLocator(integer=True))
    axs[1].grid(True)



    axs[2].plot(samples, stats.mean[13], color='purple')
    axs[2].fill_between(samples, stats.lower[13], stats.upper[13], facecolor='purple', alpha=0.5)
    axs[2].set_xlabel("samples [milions]", fontweight='bold')
    axs[2].set_ylabel("target magnitude", fontweight='bold')
    axs[2].set_xlim(left=0)
    
    axs[2].xaxis.set_major_locator(MaxNLocator(integer=True))
    axs[2].yaxis.set_major_locator(MaxNLocator(integer=True))
    axs[2].grid(True)


    axs[3].plot(samples, stats.mean[14], color='purple')
    axs[3].fill_between(samples, stats.lower[14], stats.upper[14], facecolor='purple', alpha=0.5)
    axs[3].set_xlabel("samples [milions]", fontweight='bold')
    axs[3].set_ylabel("target magnitude variance", fontweight='bold')
    axs[3].set_xlim(left=0)
    
    axs[3].xaxis.set_major_locator(MaxNLocator(integer=True))
    axs[3].yaxis.set_major_locator(MaxNLocator(integer=True))
    axs[3].grid(True)


    fig.tight_layout()
    plt.savefig(output_path + output_prefix + "_intrinsic_motivation.png", dpi = 300)




    #plot summary

    plt.clf()

    fig, axs = plt.subplots(4, 1, figsize=(10, 10))

    axs[0].plot(samples, stats.mean[3], color='deepskyblue')
    axs[0].fill_between(samples, stats.lower[3], stats.upper[3], facecolor='deepskyblue', alpha=0.5)
    axs[0].set_xlabel("samples [milions]", fontweight='bold')
    axs[0].set_ylabel("score", fontweight='bold')
    axs[0].set_xlim(left=0) 
    
    axs[0].xaxis.set_major_locator(MaxNLocator(integer=True))
    axs[0].yaxis.set_major_locator(MaxNLocator(integer=True))
    axs[0].grid(True)

    axs[1].plot(samples, stats.mean[4], color='blue')
    axs[1].fill_between(samples, stats.lower[4], stats.upper[4], facecolor='blue', alpha=0.5)
    axs[1].set_xlabel("samples [milions]", fontweight='bold')
    axs[1].set_ylabel("external reward", fontweight='bold')
    axs[1].set_xlim(left=0) 
    
    axs[1].xaxis.set_major_locator(MaxNLocator(integer=True))
    axs[1].yaxis.set_major_locator(MaxNLocator(integer=True))
    axs[1].grid(True)


    axs[2].plot(samples, sm, color='blueviolet')
    axs[2].fill_between(samples, sl, su, facecolor='blueviolet', alpha=0.5)
    axs[2].set_xlabel("samples [milions]", fontweight='bold')
    axs[2].set_ylabel("intrinsic motivation", fontweight='bold')
    axs[2].set_xlim(left=0) 
    
    axs[2].xaxis.set_major_locator(MaxNLocator(integer=True))
    axs[2].yaxis.set_major_locator(MaxNLocator(integer=True))
    axs[2].grid(True)

    axs[3].plot(samples, stats.extended_mean[0], color='purple')
    axs[3].fill_between(samples, stats.extended_lower[0], stats.extended_upper[0], facecolor='purple', alpha=0.5)
    axs[3].set_xlabel("samples [milions]", fontweight='bold')
    axs[3].set_ylabel("explored rooms count", fontweight='bold')
    axs[3].set_xlim(left=0) 
    
    axs[3].xaxis.set_major_locator(MaxNLocator(integer=True))
    axs[3].yaxis.set_major_locator(MaxNLocator(integer=True))
    axs[3].grid(True)

   
    fig.tight_layout()
    plt.savefig(output_path + output_prefix + "_summary.png", dpi = 300)
