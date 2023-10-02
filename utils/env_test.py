import numpy
import time
import RLAgents
import gymnasium as gym




def WrapperNone(env):
    return env

envs_count = 128
#envs = RLAgents.MultiEnvSeq("MontezumaRevengeNoFrameskip-v4", RLAgents.WrapperCommon, envs_count)
envs = RLAgents.MultiEnvSeq("MontezumaRevengeNoFrameskip-v4", RLAgents.WrapperCommon, envs_count)

for i in range(envs_count):
    envs.reset(i)


print("starting")
fps = 0.0

while True:
    time_start = time.time()

    action = numpy.random.randint(0, 18, envs_count)

    

    states, reward, dones, _, _ = envs.step(action)

    dones_ = numpy.where(dones)[0]
    
    for d in dones_:
        envs.reset(d)
    
    time_stop = time.time()

    k   = 0.9
    fps = k*fps + (1.0 - k)*1.0/(time_stop - time_start)
    print(fps)
