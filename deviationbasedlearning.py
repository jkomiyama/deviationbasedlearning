# -*- coding: utf-8 -*-

import numpy as np
import matplotlib
try:
    from google.colab import files
except: # https://stackoverflow.com/questions/35737116/runtimeerror-invalid-display-variable
    matplotlib.use('agg')
import matplotlib.pyplot as plt
from scipy.spatial import distance
from scipy import stats
from scipy.stats import truncnorm
import copy
import pickle

run_full = True # Test (small) or full (large)
print(f"run_full = {run_full}")
save_img = True
def is_colab():
    try:
        from google.colab import files
        return True
    except:
        return False

Figsize = (6,4)
plt.rcParams["font.size"] = 14
plt.rcParams["axes.linewidth"] = 1
plt.rcParams["lines.linewidth"] = 2
plt.rcParams["figure.subplot.bottom"] = 0.14

COLOR_MODE3 = "tab:blue"
COLOR_MODE2 = "tab:red"
COLOR_TOTAL2 = COLOR_TOTAL3 = "black"
COLOR_FOLLOW2 = COLOR_FOLLOW3 = "#cc4c0b"
COLOR_NONFOLLOW2 = COLOR_NONFOLLOW3 = "navy"
COLOR_FENCE2 = COLOR_FENCE3 = "tab:green"

linestyle_tuple = {
     'loosely dotted':        (0, (1, 10)),
     'dotted':                (0, (1, 1)),
     'densely dotted':        (0, (1, 1)),

     'loosely dashed':        (0, (5, 10)),
     'dashed':                (0, (5, 5)),
     'densely dashed':        (0, (5, 1)),

     'loosely dashdotted':    (0, (3, 10, 1, 10)),
     'dashdotted':            (0, (3, 5, 1, 5)),
     'densely dashdotted':    (0, (3, 1, 1, 1)),

     'dashdotdotted':         (0, (3, 5, 1, 5, 1, 5)),
     'loosely dashdotdotted': (0, (3, 10, 1, 10, 1, 10)),
     'densely dashdotdotted': (0, (3, 1, 1, 1, 1, 1))
     }
LINESTYLE_MODE3 = "solid"
#COLOR_MODE2 = linestyle_tuple["densely dashed"]
LINESTYLE_MODE1 = linestyle_tuple["densely dashdotdotted"]
LINESTYLE_MODE2 = "dashdot"
LINESTYLE_TOTAL2 = LINESTYLE_TOTAL3 = "solid"
LINESTYLE_FOLLOW2 = LINESTYLE_FOLLOW3 = "dashed"
LINESTYLE_MODE2EVE = linestyle_tuple["densely dashed"]
LINESTYLE_MODE2MYOPIC = "dashed"
#LINESTYLE_GREEDY = linestyle_tuple["densely dotted"]
LINESTYLE_NONFOLLOW2 = LINESTYLE_NONFOLLOW3 = linestyle_tuple["densely dotted"]
#LINESTYLE_ROONEY = "dashdot"
LINESTYLE_FENCE2 = LINESTYLE_FENCE3 = linestyle_tuple["densely dashdotted"]
#LINESTYLE_ROONEY_SWITCH = linestyle_tuple["densely dashdotted"]
#LINESTYLE_ROONEY_GREEDY = linestyle_tuple["densely dotted"]

#COLOR_CS_UCB = "navy"
#COLOR_HYBRID = "tab:orange"
#COLOR_CS_HYBRID = "#cc4c0b" 
#COLOR_ROONEY = "tab:blue"
#COLOR_ROONEY_SWITCH = "tab:green" 
#COLOR_ROONEY_GREEDY = "tab:red"

def my_show():
    try:
        from google.colab import files
        plt.show()
    except:
        pass
def colab_save(filename):
    try:
        from google.colab import files
        files.download(filename)  
    except:
        pass

#a, b = -1., 1.
if run_full:
    T = 10000
    Runnum = 5000 
else:
    T = 1000
    Runnum = 5

from joblib import Parallel, delayed
import multiprocessing as mp
print("Number of processors: ", mp.cpu_count())
NUM_PARALLEL = int(mp.cpu_count() * 0.85)
print(f"NUM_PARALLEL = {NUM_PARALLEL}")

# Binary signal (natural policy)
def round2(theta, l, u, rng):
    z = rng.normal()
    x = rng.normal()
    m = (l+u)/2
    if x*m + z > 0: # recommend arm +1
        At = 1
        ez = truncnorm.stats(-x*m, -x*m+10000, moments='mvsk')[0] #mvsk = mean,variance,stddev,kurtosis
    else: # recommends arm -1
        At = -1
        ez = truncnorm.stats(-10000-x*m, -x*m, moments='mvsk')[0]
    if x*theta + ez > 0: # user choose arm 1
        It = 1
        if x>0:
            l_new = max(l, -ez/x)
            u_new = u
        else:
            l_new = l
            u_new = min(u, -ez/x)            
    else: # user choose arm -1
        It = -1
        if x>0:
            l_new = l
            u_new = min(u, -ez/x)
        else:
            l_new = max(l, -ez/x)
            u_new = u
    # regret increase
    if x*theta + z > 0 and It == -1:
        reg = x*theta + z
    elif x*theta + z < 0 and It == 1:
        reg = - ( x*theta + z )
    else:
        reg = 0
    updated = (l_new, u_new) != (l, u)
    # shrink of the confidence bound
    update_ratio = - np.log( (u_new - l_new) / (u - l) )
    non_followed = At != It
    return (l_new, u_new, reg, updated, update_ratio, non_followed)

# Binary signal, myopic
def round2_myopic(theta, l, u, rng):
    z = rng.normal()
    x = rng.normal()
    m = (l+u)/2
    theta_vals = np.array([l + i*(u-l)/50 for i in range(50+1)]) # グリッド数は偶数推奨 (mを入れるため)
    def get_user_regret(c): 
        zst = -x*c
        phi = stats.norm.pdf(zst)
        Phi = stats.norm.cdf(zst)
        if x>=0:
            u_temp, l_temp = u, l
        else:
            u_temp, l_temp = l, u            
        #mは同じ
        if x*l_temp <= -phi/(1-Phi) <= phi/Phi <= x*u_temp: # Case 1
            temp = x*u_temp*u_temp + (1/x)*phi*phi/(Phi*(1-Phi))
            #print("case 1")
            #return (x*m*(1-Phi)+phi, 4)
            return (1/(2*(u_temp-l_temp)) * temp, 1) 
        elif -phi/(1-Phi) <= x*l_temp <= phi/Phi <= x*u_temp: # Case 2
            temp = x*u_temp*u_temp*Phi + phi*phi/(Phi*x) - 2 * phi * u_temp
            #print("case 2")
            return (x*m*(1-Phi)+phi+1/(2*(u_temp-l_temp)) * temp, 2)
        elif x*l_temp <= -phi/(1-Phi) <= x*u_temp <= phi/Phi : # Case 3
            temp = x*l_temp*l_temp*(1-Phi) + phi*phi/((1-Phi)*x) + 2 * phi * l_temp
            #print("case 3")
            return (x*m*(1-Phi)+phi+1/(2*(u_temp-l_temp)) * temp, 3)
        else: # Case 4
            #print("case 4")
            return (x*m*(1-Phi)+phi, 4)

    # optimizing threshold
    c_cands = []
    for c in theta_vals:
        val,case = get_user_regret(c)
        c_cands.append((val, case, c))
    c_best = sorted(c_cands)[-1][-1]
    case_best = sorted(c_cands)[-1][1]
    #print(f"case = {sorted(c_cands)[0][1]} c_best = {c_best} l,m,u={l},{m},{u}")

    if x*m + z > 0: # recommend arm +1
        At_straight = 1
        ez_straight = truncnorm.stats(-x*m, -x*m+10000, moments='mvsk')[0] #mvsk = mean,variance,stddev,kurtosis
    else: # recommends arm -1
        At_straight = -1
        ez_straight = truncnorm.stats(-10000-x*m, -x*m, moments='mvsk')[0]

    if x*c_best + z > 0: # recommend arm +1
        At = 1
        ez = truncnorm.stats(-x*c_best, 10000, moments='mvsk')[0] #mvsk = mean,variance,stddev,kurtosis
    else: # recommends arm -1
        At = -1
        ez = truncnorm.stats(-10000, -x*c_best, moments='mvsk')[0]
    if x*theta + ez > 0: # user choose arm 1
        It = 1
        if x>0:
            l_new = max(l, -ez/x)
            u_new = u
        else:
            l_new = l
            u_new = min(u, -ez/x)            
    else: # user choose arm -1
        It = -1
        if x>0:
            l_new = l
            u_new = min(u, -ez/x)
        else:
            l_new = max(l, -ez/x)
            u_new = u
    # regret increase
    if x*theta + z > 0 and It == -1:
        reg = x*theta + z
    elif x*theta + z < 0 and It == 1:
        reg = - ( x*theta + z )
    else:
        reg = 0
    updated = (l_new, u_new) != (l, u)
    # shrink of the confidence bound
    update_ratio = - np.log( (u_new - l_new) / (u - l) )
    non_followed = At != It
    if At != At_straight:
        print(f"difference between straightforward and myopic recommendation")
    if (x*theta + ez_straight) * (x*theta + ez) < 0:
        print(f"difference between straightforward and myopic user decision")
    return (l_new, u_new, reg, updated, update_ratio, non_followed, case_best)


# Binary signal, explorative strategy
def round2_explore(theta, l, u, rng):
    z = rng.normal()
    x = rng.normal()
    m = (l+u)/2 #m
    def get_threshold(Z):
        if Z < 0:
            return -get_threshold(-Z)
        else: # Z >= 0
            l_Z = -100
            u_Z = 0
            EPS = 0.0000001
            while u_Z - l_Z > EPS:
                m_Z = (l_Z + u_Z) / 2.0
                cur_Z = truncnorm.stats(-100, m_Z, moments='mvsk')[0]
                if cur_Z > -Z:
                    u_Z = m_Z
                else:
                    l_Z = m_Z
            print(f"final -Z, -cur_Z = {truncnorm.stats(-100, m_Z, moments='mvsk')[0]}")
            return m_Z
    threshold = get_threshold(x*m)
    #exploreferent = np.abs(z) > np.abs(threshold) #exploreferent side これなんだっけ・・・
    if z > threshold: # recommend arm +1
        At = 1
    else: # recommends arm -1
        At = -1
    if At == 1:
        ez = truncnorm.stats(threshold, 100, moments='mvsk')[0]
        if x*theta + ez > 0:
            It = 1
        else:
            It = -1
        #l_new, u_new = l, u
        if x * It > 0:
            l_new = max(l, -ez/x)
            u_new = u
        else:
            l_new = l
            u_new = min(u, -ez/x) 
    else:
        if x * (theta - m) > 0:
            It = 1
        else:
            It = -1
        if theta > m:
            l_new = m
            u_new = u
        else:
            l_new = l
            u_new = m
    # regret increase
    if x*theta + z > 0 and It == -1:
        reg = x*theta + z
    elif x*theta + z < 0 and It == 1:
        reg = - ( x*theta + z )
    else:
        reg = 0
    updated = (l_new, u_new) != (l, u)
    # shrink of the confidence bound
    print(f"l, theta, u = {l}, {theta}, {u}")
    update_ratio = - np.log( (u_new - l_new) / (u - l) )
    non_followed = At != It
    return (l_new, u_new, reg, updated, update_ratio, non_followed)

# Ternary signal
def round3(theta, l, u, eps, rng):
    z = rng.normal()
    x = rng.normal()
    m = (l+u)/2
    if abs(x*m + z) < eps: # recommends 0
        ez = truncnorm.stats(-x*m-eps, -x*m+eps, moments='mvsk')[0]
        At = 0
    elif x*m + z > 0: # recommends arm 1
        ez = truncnorm.stats(-x*m+eps, -x*m+10000, moments='mvsk')[0]
        At = 1
    else: # recommends arm -1
        ez = truncnorm.stats(-10000-x*m, -x*m-eps, moments='mvsk')[0]
        At = -1
    if x*theta + ez > 0: # user chooses arm 1
        It = 1
        if x>0:
            l_new = max(l, -ez/x)
            u_new = u
        else:
            l_new = l
            u_new = min(u, -ez/x)
    else: # user chooses arm -1
        It = -1
        if x>0:
            l_new = l
            u_new = min(u, -ez/x)
        else:
            l_new = max(l, -ez/x)
            u_new = u

    # regret
    if x*theta + z > 0 and It != 1:
        reg = x*theta + z
    elif x*theta + z < 0 and It != -1:
        reg = - ( x*theta + z )
    else:
        reg = 0

    updated = (l_new, u_new) != (l, u)
    update_ratio = - np.log( (u_new - l_new) / (u - l) )
    non_followed = At != It
    onthefence = At == 0
    return (l_new, u_new, reg, updated, update_ratio, non_followed, onthefence)

def run_once(mode, rs):
    rng = np.random.RandomState(rs)
    Regs = np.zeros(T) # Regret incurred each round and each run
    follow_update_num = np.zeros(T) # num of updates by following
    nonfollow_update_num = np.zeros(T) # num of updates by deviation
    fence_update_num = np.zeros(T) # num of updates by on the fence
    follow_update = np.zeros(T) # amount of updates by following
    nonfollow_update = np.zeros(T) # amount of updates by deviation
    fence_update = np.zeros(T) # amount of updates by on the fence
    
    l, u = -1, 1
    theta = -1 + 2 * rng.uniform()
    print(f"theta = {theta}")
    tot_updated, tot_non_followed = 0, 0
    tot_case = [0,0,0,0] 
    for t in range(T):
        if mode == "mode3":
            eps = 0.25 * (u-l) # on the fence width
            l, u, reg, updated, update_ratio, non_followed, onthefence = round3(theta, l, u, eps, rng)
        elif mode == "mode2_explore":
            EPS = 0.0000001
            r = rng.random()
            if u-l > 1./np.sqrt(T): 
                l, u, reg, updated, update_ratio, non_followed = round2_explore(theta, l, u, rng)
            else: #stop on the fence
                l, u, reg, updated, update_ratio, non_followed, case = round2_myopic(theta, l, u, rng)
                #l, u, reg, updated, update_ratio, non_followed = round2(theta, l, u, rng)
            onthefence = False
        elif mode == "mode2_myopic": #mode2
            l, u, reg, updated, update_ratio, non_followed, case = round2_myopic(theta, l, u, rng)
            tot_case[case-1] += 1
            onthefence = False
            #print(f"l, u, reg, updated, update_ratio, non_followed = {l, u, reg, updated, update_ratio, non_followed}")
        else: #mode2
            #round2_myopic(theta, l, u) 
            l, u, reg, updated, update_ratio, non_followed = round2(theta, l, u, rng)
            onthefence = False
            #print(f"l, u, reg, updated, update_ratio, non_followed = {l, u, reg, updated, update_ratio, non_followed}")
        if updated:
            if onthefence:
                fence_update_num[t] = 1
                fence_update[t] = update_ratio
            elif non_followed:
                nonfollow_update_num[t] = 1
                nonfollow_update[t] = update_ratio
            else:
                follow_update_num[t] = 1
                follow_update[t] = update_ratio
        Regs[t] = reg
    result = (Regs, follow_update_num, nonfollow_update_num, fence_update_num, follow_update, nonfollow_update, fence_update)
    return result
    #print(f"run = {run}, (l, u)=({l},{u}) d={u-l} tot_case={tot_case}")

#これは1モードごとに計算
def main(mode):
    print(f"mode = {mode}")
    rss = np.random.randint(np.iinfo(np.int32).max, size=Runnum)
    #print(f"rss = {rss}")
    result_list = Parallel(n_jobs=NUM_PARALLEL)( [delayed(run_once)(mode=mode, rs=rss[r]) for r in range(Runnum)] ) 

    Regs = np.zeros((T, Runnum)) # Regret incurred each round and each run
    follow_update_num = np.zeros((T, Runnum)) # num of updates by following
    nonfollow_update_num = np.zeros((T, Runnum)) # num of updates by deviation
    fence_update_num = np.zeros((T, Runnum)) # num of updates by on the fence
    follow_update = np.zeros((T, Runnum)) # amount of updates by following
    nonfollow_update = np.zeros((T, Runnum)) # amount of updates by deviation
    fence_update = np.zeros((T, Runnum)) # amount of updates by on the fence
    for run, result in enumerate(result_list):
        Regs[:, run], follow_update_num[:, run], nonfollow_update_num[:, run], fence_update_num[:, run],\
           follow_update[:, run], nonfollow_update[:, run], fence_update[:, run] = result[0], result[1], result[2], result[3], result[4], result[5], result[6]

    #print(f"Regs = {Regs}")
    results = (Regs, follow_update_num, nonfollow_update_num, fence_update_num, follow_update, nonfollow_update, fence_update)
    return results

# Unitary (not used)
def single1(x, z, l, u, theta): 
    m = (l+u)/2
    if (x * theta) * (x * theta + z) < 0:
        # positive regret
        reg = np.abs(x * theta + z)
    else:
        # no regret
        reg = 0
    if x*theta > 0: # user choose arm 1
        It = 1
        if x>0:
            l_new = max(l, 0)
            u_new = u
        else:
            l_new = l
            u_new = min(u, 0)            
    else: # user choose arm -1
        It = -1
        if x>0:
            l_new = l
            u_new = min(u, 0)
        else:
            l_new = max(l, 0)
            u_new = u
    
    shrink = (u-l) - (u_new-l_new)
    if u_new < l_new:
        print(f"Error: range violation")
    return reg,shrink

def single2(x, z, l, u, theta): 
    m = (l+u)/2
    #z = theta
    if x*m + z > 0: # recommend arm +1
        At = 1
        ez = truncnorm.stats(-x*m, -x*m+10000, moments='mvsk')[0] #mvsk = mean,variance,stddev,kurtosis
    else: # recommends arm -1
        At = -1
        ez = truncnorm.stats(-10000-x*m, -x*m, moments='mvsk')[0]
    if x*theta + ez > 0: # user choose arm 1
        It = 1
        if x>0:
            l_new = max(l, -ez/x)
            u_new = u
        else:
            l_new = l
            u_new = min(u, -ez/x)            
    else: # user choose arm -1
        It = -1
        if x>0:
            l_new = l
            u_new = min(u, -ez/x)
        else:
            l_new = max(l, -ez/x)
            u_new = u
    # regret increase
    if x*theta + z > 0 and It == -1:
        reg = x*theta + z
    elif x*theta + z < 0 and It == 1:
        reg = - ( x*theta + z )
    else:
        reg = 0
    shrink = (u-l) - (u_new-l_new)
    #print(f"single 2 x={x},l={l},u={u},u_new={u_new},l_new={l_new},shrink={shrink}")
    if u_new < l_new:
        print(f"Error: range violation")
    return reg,shrink

def single2_myopic(x, z, l, u, theta): 
    m = (l+u)/2

    theta_vals = np.array([l + i*(u-l)/50 for i in range(50+1)]) 
    #EPS = 0.000001
    def get_user_regret(c): 
        zst = -x*c
        phi = stats.norm.pdf(zst)
        Phi = stats.norm.cdf(zst)
        if x>=0:
            u_temp, l_temp = u, l
        else:
            u_temp, l_temp = l, u            
        if x*l_temp <= -phi/(1-Phi) <= phi/Phi <= x*u_temp: # Case 1
            temp = x*u_temp*u_temp + (1/x)*phi*phi/(Phi*(1-Phi))
            #print("case 1")
            #return (x*m*(1-Phi)+phi, 4)
            return (1/(2*(u_temp-l_temp)) * temp, 1) 
        elif -phi/(1-Phi) <= x*l_temp <= phi/Phi <= x*u_temp: # Case 2
            temp = x*u_temp*u_temp*Phi + phi*phi/(Phi*x) - 2 * phi * u_temp
            #print("case 2")
            return (x*m*(1-Phi)+phi+1/(2*(u_temp-l_temp)) * temp, 2)
        elif x*l_temp <= -phi/(1-Phi) <= x*u_temp <= phi/Phi : # Case 3
            temp = x*l_temp*l_temp*(1-Phi) + phi*phi/((1-Phi)*x) + 2 * phi * l_temp
            #print("case 3")
            return (x*m*(1-Phi)+phi+1/(2*(u_temp-l_temp)) * temp, 3)
        else: # Case 4
            #print("case 4")
            return (x*m*(1-Phi)+phi, 4)

    c_cands = []
    for c in theta_vals:
        val,case = get_user_regret(c)
        c_cands.append((val, case, c))
    c_best = sorted(c_cands)[-1][-1]
    case_best = sorted(c_cands)[-1][1]
    #if case_best != 4:
    if c_best != m and case_best != 4:
        print(f"case_best = {case_best} c_best = {c_best} l,m,u={l},{m},{u}")
    if x*c_best + z > 0: # recommend arm +1
        At = 1
        ez = truncnorm.stats(-x*c_best, 10000, moments='mvsk')[0] #mvsk = mean,variance,stddev,kurtosis
    else: # recommends arm -1
        At = -1
        ez = truncnorm.stats(-10000, -x*c_best, moments='mvsk')[0]
    if x*theta + ez > 0: # user choose arm 1
        It = 1
        if x>0:
            l_new = max(l, -ez/x)
            u_new = u
        else:
            l_new = l
            u_new = min(u, -ez/x)            
    else: # user choose arm -1
        It = -1
        if x>0:
            l_new = l
            u_new = min(u, -ez/x)
        else:
            l_new = max(l, -ez/x)
            u_new = u
    # regret increase
    if x*theta + z > 0 and It == -1:
        reg = x*theta + z
    elif x*theta + z < 0 and It == 1:
        reg = - ( x*theta + z )
    else:
        reg = 0

    shrink = (u-l) - (u_new-l_new)
    #print(f"single 2 x={x},l={l},u={u},u_new={u_new},l_new={l_new},shrink={shrink}")
    if u_new < l_new:
        print(f"Error: range violation")
    return reg,shrink

def single2_explore(x, z, l, u, theta): 
    m = (l+u)/2
    def get_threshold(Z):
        if Z < 0:
            return -get_threshold(-Z)
        else: # Z >= 0
            l_Z = -100
            u_Z = 0
            EPS = 0.0000001
            while u_Z - l_Z > EPS:
                m_Z = (l_Z + u_Z) / 2.0
                cur_Z = truncnorm.stats(-100, m_Z, moments='mvsk')[0]
                if cur_Z > -Z:
                    u_Z = m_Z
                else:
                    l_Z = m_Z
            #print(f"final -Z, -cur_Z = {truncnorm.stats(-100, m_Z, moments='mvsk')[0]}")
            return m_Z
    threshold = get_threshold(x*m)
    if z > threshold: # recommend arm +1
        At = 1
    else: # recommends arm -1
        At = -1
    if At == 1:
        ez = truncnorm.stats(threshold, 100, moments='mvsk')[0]
        if x*theta + ez > 0:
            It = 1
        else:
            It = -1
        #l_new, u_new = l, u
        if x * It > 0:
            l_new = max(l, -ez/x)
            u_new = u
        else:
            l_new = l
            u_new = min(u, -ez/x) 
    else:
        if x * (theta - m) > 0:
            It = 1
        else:
            It = -1
        if theta > m:
            l_new = m
            u_new = u
        else:
            l_new = l
            u_new = m
    # regret increase
    if x*theta + z > 0 and It == -1:
        reg = x*theta + z
    elif x*theta + z < 0 and It == 1:
        reg = - ( x*theta + z )
    else:
        reg = 0
    shrink = (u-l) - (u_new-l_new)
    #print(f"single 2 x={x},l={l},u={u},u_new={u_new},l_new={l_new},shrink={shrink}")
    if u_new < l_new:
        print(f"Error: range violation")
    return reg,shrink
   

def single3(x, z, l, u, theta): 
    m = (l+u)/2
    #z = theta
#    eps = 0.75 * (u-l) # on the fence width
    eps = 0.25 * (u-l) # on the fence width
    if abs(x*m + z) < eps: # recommends 0
        ez = truncnorm.stats(-x*m-eps, -x*m+eps, moments='mvsk')[0]
        At = 0
    elif x*m + z > 0: # recommends arm 1
        ez = truncnorm.stats(-x*m+eps, -x*m+10000, moments='mvsk')[0]
        At = 1
    else: # recommends arm -1
        ez = truncnorm.stats(-10000-x*m, -x*m-eps, moments='mvsk')[0]
        At = -1
    if x*theta + ez > 0: # user chooses arm 1
        It = 1
        if x>0:
            l_new = max(l, -ez/x)
            u_new = u
        else:
            l_new = l
            u_new = min(u, -ez/x)
    else: # user chooses arm -1
        It = -1
        if x>0:
            l_new = l
            u_new = min(u, -ez/x)
        else:
            l_new = max(l, -ez/x)
            u_new = u

    if x*theta + z > 0 and It != 1:
        reg = x*theta + z
    elif x*theta + z < 0 and It != -1:
        reg = - ( x*theta + z )
    else:
        reg = 0
    shrink = (u-l) - (u_new-l_new)
    #print(f"single 3 x={x},l={l},u={u},u_new={u_new},l_new={l_new},shrink={shrink}")
    if u_new < l_new:
        print(f"Error: range violation")
    return reg,shrink

# subroutine 
# e.g. 1,2,3,4 -> 1,3,6,10 
def my_conv(anarray):
    if len(anarray.shape)>1: #assuming 1 or 2 dim
        T = anarray.shape[0]
        R = anarray.shape[1]
        temp = np.zeros(R)
        ret_array = np.zeros( (T, R) )
        for t in range(T):
            temp += anarray[t,:]
            ret_array[t,:] = temp
    else:
        T = anarray.shape[0]
        temp = 0 
        ret_array = np.zeros(T)
        for t in range(T):
            temp += anarray[t]
            ret_array[t] = temp
    return ret_array

def compare_singleround_once(l, u, GRID_SIZE, rs):
    rng = np.random.RandomState(rs)
    x = rng.normal() # + 10
    z = rng.normal()
    #print(f"x={x}, z={z}")
    Reg1_vec = np.zeros((GRID_SIZE)) #unitary short_term value
    Reg2_vec = np.zeros((GRID_SIZE)) #binary straight short_term value
    Reg2myopic_vec = np.zeros((GRID_SIZE)) #binary myopic short_term value
    Reg2explore_vec = np.zeros((GRID_SIZE)) #binary explore short_term value
    Reg3_vec = np.zeros((GRID_SIZE)) #ternary short_term value
    Shrink1_vec = np.zeros((GRID_SIZE)) #unitary long_term value
    Shrink2_vec = np.zeros((GRID_SIZE)) #binary straight long_term value
    Shrink2myopic_vec = np.zeros((GRID_SIZE)) #binary myopic long_term value
    Shrink2explore_vec = np.zeros((GRID_SIZE)) #binary explore long_term value
    Shrink3_vec = np.zeros((GRID_SIZE)) #ternary long_term value
    for thetai in range(GRID_SIZE):
        theta = l + (u-l)*thetai/(GRID_SIZE-1)
        diff1 = single1(x, z, l, u, theta)
        diff2 = single2(x, z, l, u, theta)
        diff2_myopic = single2_myopic(x, z, l, u, theta)
        diff2_explore = single2_explore(x, z, l, u, theta)
        diff3 = single3(x, z, l, u, theta)
        Reg1_vec[thetai] += diff1[0]
        Shrink1_vec[thetai] += diff1[1]
        Reg2_vec[thetai] += diff2[0]
        Shrink2_vec[thetai] += diff2[1]
        Reg2myopic_vec[thetai] += diff2_myopic[0]
        Shrink2myopic_vec[thetai] += diff2_myopic[1]
        Reg2explore_vec[thetai] += diff2_explore[0]
        Shrink2explore_vec[thetai] += diff2_explore[1]
        Reg3_vec[thetai] += diff3[0]
        Shrink3_vec[thetai] += diff3[1]
    return (Reg1_vec, Shrink1_vec, Reg2_vec, Shrink2_vec, Reg2myopic_vec, Shrink2myopic_vec,\
        Reg2explore_vec, Shrink2explore_vec, Reg3_vec, Shrink3_vec)

def compare_singleround():
    GRID_SIZE = 20
    if run_full:
        TRIAL_NUM = 10000
    else:
        TRIAL_NUM = 1000    
    Reg1_mat = np.zeros((GRID_SIZE, GRID_SIZE)) 
    Reg2_mat = np.zeros((GRID_SIZE, GRID_SIZE)) 
    Reg2myopic_mat = np.zeros((GRID_SIZE, GRID_SIZE)) 
    Reg2explore_mat = np.zeros((GRID_SIZE, GRID_SIZE)) 
    Reg3_mat = np.zeros((GRID_SIZE, GRID_SIZE)) 
    Shrink1_mat = np.zeros((GRID_SIZE, GRID_SIZE))
    Shrink2_mat = np.zeros((GRID_SIZE, GRID_SIZE))
    Shrink2myopic_mat = np.zeros((GRID_SIZE, GRID_SIZE))
    Shrink2explore_mat = np.zeros((GRID_SIZE, GRID_SIZE)) 
    Shrink3_mat = np.zeros((GRID_SIZE, GRID_SIZE)) 
    ls, us = [], []

    for ui in range(GRID_SIZE):
        u = 0.5 # u - 1*(ui+1)/GRID_SIZE
        l = u - 1.5*(ui+1)/GRID_SIZE #0.5  
        ls.append(l)
        us.append(u)
        rss = np.random.randint(np.iinfo(np.int32).max, size=TRIAL_NUM)
        #print(f"rss = {rss}")
        result_list = Parallel(n_jobs=NUM_PARALLEL)( [delayed(compare_singleround_once)(l=l, u=u, GRID_SIZE=GRID_SIZE, rs=rss[r]) for r in range(TRIAL_NUM)] ) 
        for temp in result_list:
            Reg1_mat[ui] += temp[0]
            Shrink1_mat[ui] += temp[1]
            Reg2_mat[ui] += temp[2]
            Shrink2_mat[ui] += temp[3]
            Reg2myopic_mat[ui] += temp[4]
            Shrink2myopic_mat[ui] += temp[5]
            Reg2explore_mat[ui] += temp[6]
            Shrink2explore_mat[ui] += temp[7]
            Reg3_mat[ui] += temp[8]
            Shrink3_mat[ui] += temp[9]
    #print(f"shape = {Shrink3_mat.shape}")
    Reg1_mat /= TRIAL_NUM
    Reg2_mat /= TRIAL_NUM
    Reg2myopic_mat /= TRIAL_NUM
    Reg2explore_mat /= TRIAL_NUM
    #print(f"shape = {Shrink3_mat.shape}")
    Reg3_mat /= TRIAL_NUM
    Shrink1_mat /= TRIAL_NUM
    Shrink2_mat /= TRIAL_NUM
    Shrink2myopic_mat /= TRIAL_NUM
    Shrink2explore_mat /= TRIAL_NUM
    Shrink3_mat /= TRIAL_NUM
    results = (np.array(ls), np.array(us),\
        np.mean(Reg1_mat, axis=1), np.mean(Shrink1_mat, axis=1),\
        np.mean(Reg2_mat, axis=1), np.mean(Shrink2_mat, axis=1),\
        np.mean(Reg2myopic_mat, axis=1), np.mean(Shrink2myopic_mat, axis=1),\
        np.mean(Reg2explore_mat, axis=1), np.mean(Shrink2explore_mat, axis=1),\
        np.mean(Reg3_mat, axis=1), np.mean(Shrink3_mat, axis=1))
    return results



# plot
def my_plot(result_mode2, result_mode2myopic, result_mode2explore, result_mode3, confidence_bound = True):
    LP, UP = 0.25, 0.75
    fig = plt.figure(figsize=Figsize)
    Regs_mode2 = my_conv(np.sum(result_mode2[0], axis=1)/Runnum)
    Regs_mode2myopic = my_conv(np.sum(result_mode2myopic[0], axis=1)/Runnum)
    Regs_mode2explore = my_conv(np.sum(result_mode2explore[0], axis=1)/Runnum)
    Regs_mode3 = my_conv(np.sum(result_mode3[0], axis=1)/Runnum)


    print(f"Binary straight (last T) = {Regs_mode2[-1]}")
    print(f"Binary myopic (last T) = {Regs_mode2myopic[-1]}")
    print(f"Binary EvE (last T) = {Regs_mode2explore[-1]}")
    print(f"Ternary (last T) = {Regs_mode3[-1]}")

    if confidence_bound:
        lower_quantile_regs_mode2 = np.quantile(my_conv(result_mode2[0]), LP, axis=1)
        upper_quantile_regs_mode2 = np.quantile(my_conv(result_mode2[0]), UP, axis=1)
        std_Regs_mode2 = np.std(my_conv(result_mode2[0]), axis=1)
        plt.errorbar([T], Regs_mode2[-1], yerr=2*std_Regs_mode2[-1]/np.sqrt(Runnum), fmt='o', capsize = 5, color = COLOR_MODE2) #2 sigma
        plt.fill_between(range(1, len(Regs_mode2)+1), lower_quantile_regs_mode2, upper_quantile_regs_mode2, alpha=0.3, color = COLOR_MODE2)

        lower_quantile_regs_mode2myopic = np.quantile(my_conv(result_mode2myopic[0]), LP, axis=1)
        upper_quantile_regs_mode2myopic = np.quantile(my_conv(result_mode2myopic[0]), UP, axis=1)
        std_Regs_mode2myopic = np.std(my_conv(result_mode2myopic[0]), axis=1)
        plt.errorbar([T], Regs_mode2myopic[-1], yerr=2*std_Regs_mode2myopic[-1]/np.sqrt(Runnum), fmt='o', capsize = 5, color = "tab:orange") #2 sigma
        plt.fill_between(range(1, len(Regs_mode2)+1), lower_quantile_regs_mode2myopic, upper_quantile_regs_mode2myopic, alpha=0.3, color = "tab:orange")

        lower_quantile_regs_mode2explore = np.quantile(my_conv(result_mode2explore[0]), LP, axis=1)
        upper_quantile_regs_mode2explore = np.quantile(my_conv(result_mode2explore[0]), UP, axis=1)
        std_Regs_mode2explore = np.std(my_conv(result_mode2explore[0]), axis=1)
        plt.errorbar([T], Regs_mode2explore[-1], yerr=2*std_Regs_mode2explore[-1]/np.sqrt(Runnum), fmt='o', capsize = 5, color = "green") #2 sigma
        plt.fill_between(range(1, len(Regs_mode2)+1), lower_quantile_regs_mode2explore, upper_quantile_regs_mode2explore, alpha=0.3, color = "green")

        lower_quantile_regs_mode3 = np.quantile(my_conv(result_mode3[0]), LP, axis=1)
        upper_quantile_regs_mode3 = np.quantile(my_conv(result_mode3[0]), UP, axis=1)
        std_Regs_mode3 = np.std(my_conv(result_mode3[0]), axis=1)
        plt.errorbar([T], Regs_mode3[-1], yerr=2*std_Regs_mode3[-1]/np.sqrt(Runnum), fmt='o', capsize = 5, color = COLOR_MODE3) #2 sigma
        plt.fill_between(range(1, len(Regs_mode3)+1), lower_quantile_regs_mode3, upper_quantile_regs_mode3, alpha=0.3, color = COLOR_MODE3)
    plt.plot(range(1, len(Regs_mode2)+1), Regs_mode2, label = "Binary (straight)", color = COLOR_MODE2, linestyle = LINESTYLE_MODE2)
    plt.plot(range(1, len(Regs_mode2myopic)+1), Regs_mode2myopic, label = "Binary (myopic)", color = "tab:orange", linestyle = LINESTYLE_MODE2MYOPIC)
    plt.plot(range(1, len(Regs_mode2explore)+1), Regs_mode2explore, label = "Binary (EvE)", color = "green", linestyle = LINESTYLE_MODE2EVE)
    plt.plot(range(1, len(Regs_mode3)+1), Regs_mode3, label = "Ternary", color = COLOR_MODE3, linestyle = LINESTYLE_MODE3)
    plt.legend()
    plt.ylabel("Regret")
    plt.xlabel("Round (t)")
    my_show()
    if save_img:
        fig.savefig("regret.pdf", dpi=fig.dpi, bbox_inches='tight')
        colab_save("regret.pdf")
    plt.clf()
    fig = plt.figure(figsize=Figsize)
    if confidence_bound:
        plt.errorbar([T], Regs_mode2[-1], yerr=2*std_Regs_mode2[-1]/np.sqrt(Runnum), fmt='o', capsize = 5, color = COLOR_MODE2) #2 sigma
        plt.fill_between(range(1, len(Regs_mode2)+1), lower_quantile_regs_mode2, upper_quantile_regs_mode2, alpha=0.3, color = COLOR_MODE2)
    plt.plot(range(1, len(Regs_mode2)+1), Regs_mode2, label = "Binary (straight)", color = COLOR_MODE2, linestyle = LINESTYLE_MODE2)
    plt.ylabel("Regret")
    plt.xlabel("Round (t)")
    my_show()
    if save_img:
        fig.savefig("regret2.pdf", dpi=fig.dpi, bbox_inches='tight')
        colab_save("regret2.pdf")

    plt.clf()
    fig = plt.figure(figsize=Figsize)
    if confidence_bound:
        plt.errorbar([T], Regs_mode2myopic[-1], yerr=2*std_Regs_mode2myopic[-1]/np.sqrt(Runnum), fmt='o', capsize = 5, color = "tab:orange") #2 sigma
        plt.fill_between(range(1, len(Regs_mode2myopic)+1), lower_quantile_regs_mode2myopic, upper_quantile_regs_mode2myopic, alpha=0.3, color = "tab:orange")
    plt.plot(range(1, len(Regs_mode2myopic)+1), Regs_mode2myopic, label = "Binary (straight)", color = "tab:orange", linestyle = LINESTYLE_MODE2)
    plt.ylabel("Regret")
    plt.xlabel("Round (t)")
    my_show()
    if save_img:
        fig.savefig("regret2myopic.pdf", dpi=fig.dpi, bbox_inches='tight')
        colab_save("regret2myopic.pdf")

    plt.clf()
    fig = plt.figure(figsize=Figsize)
    if confidence_bound:
        plt.errorbar([T], Regs_mode2explore[-1], yerr=2*std_Regs_mode2explore[-1]/np.sqrt(Runnum), fmt='o', capsize = 5, color = "green") #2 sigma
        plt.fill_between(range(1, len(Regs_mode2explore)+1), lower_quantile_regs_mode2explore, upper_quantile_regs_mode2explore, alpha=0.3, color = "green")
    plt.plot(range(1, len(Regs_mode2explore)+1), Regs_mode2explore, label = "Binary", color = "green", linestyle = LINESTYLE_MODE2)
    plt.ylabel("Regret")
    plt.xlabel("Round (t)")
    my_show()
    if save_img:
        fig.savefig("regret2explore.pdf", dpi=fig.dpi, bbox_inches='tight')
        colab_save("regret2explore.pdf")

    plt.clf()
    fig = plt.figure(figsize=Figsize)
    if confidence_bound:
        plt.errorbar([T], Regs_mode3[-1], yerr=2*std_Regs_mode3[-1]/np.sqrt(Runnum), fmt='o', capsize = 5, color = COLOR_MODE3) #2 sigma
        plt.fill_between(range(1, len(Regs_mode3)+1), lower_quantile_regs_mode3, upper_quantile_regs_mode3, alpha=0.3, color = COLOR_MODE3)
    plt.plot(range(1, len(Regs_mode3)+1), Regs_mode3, label = "Ternary", color = COLOR_MODE3, linestyle = LINESTYLE_MODE3)
    plt.ylabel("Regret")
    plt.xlabel("Round (t)")
    my_show()
    if save_img:
        fig.savefig("regret3.pdf", dpi=fig.dpi, bbox_inches='tight')
        colab_save("regret3.pdf")
    plt.clf()

    fig = plt.figure(figsize=Figsize)
    follow_update_num_mode2 = my_conv(np.sum(result_mode2[1], axis=1)/Runnum)
    nonfollow_update_num_mode2 = my_conv(np.sum(result_mode2[2], axis=1)/Runnum)
    total_update_num_mode2 = follow_update_num_mode2 + nonfollow_update_num_mode2
    if confidence_bound:
        lower_quantile_total_update_num_mode2 = np.quantile(my_conv(result_mode2[1]+result_mode2[2]), LP, axis=1)
        upper_quantile_total_update_num_mode2 = np.quantile(my_conv(result_mode2[1]+result_mode2[2]), UP, axis=1)
        std_total_update_num_mode2 = np.std(my_conv(result_mode2[1]+result_mode2[2]), axis=1)
        plt.errorbar([T], total_update_num_mode2[-1], yerr=2*std_total_update_num_mode2[-1]/np.sqrt(Runnum), fmt='o', capsize = 5, color = COLOR_TOTAL2) #2 sigma
        plt.fill_between(range(1, len(Regs_mode2)+1), lower_quantile_total_update_num_mode2, upper_quantile_total_update_num_mode2, alpha=0.3, color = COLOR_TOTAL2)
        lower_quantile_follow_update_num_mode2 = np.quantile(my_conv(result_mode2[1]), LP, axis=1)
        upper_quantile_follow_update_num_mode2 = np.quantile(my_conv(result_mode2[1]), UP, axis=1)
        std_follow_update_num_mode2 = np.std(my_conv(result_mode2[1]), axis=1)
        plt.errorbar([T], follow_update_num_mode2[-1], yerr=2*std_follow_update_num_mode2[-1]/np.sqrt(Runnum), fmt='o', capsize = 5, color = COLOR_FOLLOW2) #2 sigma
        plt.fill_between(range(1, len(Regs_mode2)+1), lower_quantile_follow_update_num_mode2, upper_quantile_follow_update_num_mode2, alpha=0.3, color = COLOR_FOLLOW2)
        std_nonfollow_update_num_mode2 = np.std(my_conv(result_mode2[2]), axis=1)
        lower_quantile_nonfollow_update_num_mode2 = np.quantile(my_conv(result_mode2[2]), LP, axis=1)
        upper_quantile_nonfollow_update_num_mode2 = np.quantile(my_conv(result_mode2[2]), UP, axis=1)
        plt.errorbar([T], nonfollow_update_num_mode2[-1], yerr=2*std_nonfollow_update_num_mode2[-1]/np.sqrt(Runnum), fmt='o', capsize = 5, color = COLOR_NONFOLLOW2) #2 sigma
        plt.fill_between(range(1, len(Regs_mode2)+1), lower_quantile_nonfollow_update_num_mode2, upper_quantile_nonfollow_update_num_mode2, alpha=0.3, color = COLOR_NONFOLLOW2)
    plt.plot(range(1, len(Regs_mode2)+1), total_update_num_mode2, label = "Total", color = COLOR_TOTAL2, linestyle = LINESTYLE_TOTAL2)
    plt.plot(range(1, len(Regs_mode2)+1), follow_update_num_mode2, label = "Obey", color = COLOR_FOLLOW2, linestyle = LINESTYLE_FOLLOW2)
    plt.plot(range(1, len(Regs_mode2)+1), nonfollow_update_num_mode2, label = "Deviate", color = COLOR_NONFOLLOW2, linestyle = LINESTYLE_NONFOLLOW2)
    plt.legend()
    plt.ylabel("Number of Updates")
    plt.xlabel("Round (t)")
    my_show()
    if save_img:
        fig.savefig("update_num2.pdf", dpi=fig.dpi, bbox_inches='tight')
        colab_save("update_num2.pdf")
    plt.clf()

    fig = plt.figure(figsize=Figsize)
    follow_update_num_mode3 = my_conv(np.sum(result_mode3[1], axis=1)/Runnum)
    nonfollow_update_num_mode3 = my_conv(np.sum(result_mode3[2], axis=1)/Runnum)
    fence_update_num_mode3 = my_conv(np.sum(result_mode3[3], axis=1)/Runnum)
    total_update_num_mode3 = follow_update_num_mode3 + nonfollow_update_num_mode3 + fence_update_num_mode3
    if confidence_bound:
        #lower_quantile_total_update_num_mode3 = np.quantile(my_conv(result_mode3[1]+result_mode3[2]+result_mode3[3]), LP, axis=1)
        #upper_quantile_total_update_num_mode3 = np.quantile(my_conv(result_mode3[1]+result_mode3[2]+result_mode3[3]), UP, axis=1)
        #std_total_update_num_mode3 = np.std(my_conv(result_mode3[1]+result_mode3[2]+result_mode3[3]), axis=1)
        #plt.errorbar([T], total_update_num_mode3[-1], yerr=2*std_total_update_num_mode3[-1]/np.sqrt(Runnum), fmt='o', capsize = 5, color = COLOR_TOTAL3) #2 sigma
        #plt.fill_between(range(1, len(Regs_mode2)+1), lower_quantile_total_update_num_mode3, upper_quantile_total_update_num_mode3, alpha=0.3, color = COLOR_TOTAL3)
        lower_quantile_follow_update_num_mode3 = np.quantile(my_conv(result_mode3[1]), LP, axis=1)
        upper_quantile_follow_update_num_mode3 = np.quantile(my_conv(result_mode3[1]), UP, axis=1)
        std_follow_update_num_mode3 = np.std(my_conv(result_mode3[1]), axis=1)
        plt.errorbar([T], follow_update_num_mode3[-1], yerr=2*std_follow_update_num_mode3[-1]/np.sqrt(Runnum), fmt='o', capsize = 5, color = COLOR_FOLLOW3) #2 sigma
        plt.fill_between(range(1, len(Regs_mode2)+1), lower_quantile_follow_update_num_mode3, upper_quantile_follow_update_num_mode3, alpha=0.3, color = COLOR_FOLLOW3)
        std_nonfollow_update_num_mode3 = np.std(my_conv(result_mode3[2]), axis=1)
        lower_quantile_nonfollow_update_num_mode3 = np.quantile(my_conv(result_mode3[2]), LP, axis=1)
        upper_quantile_nonfollow_update_num_mode3 = np.quantile(my_conv(result_mode3[2]), UP, axis=1)
        plt.errorbar([T], nonfollow_update_num_mode3[-1], yerr=2*std_nonfollow_update_num_mode3[-1]/np.sqrt(Runnum), fmt='o', capsize = 5, color = COLOR_NONFOLLOW3) #2 sigma
        plt.fill_between(range(1, len(Regs_mode2)+1), lower_quantile_nonfollow_update_num_mode3, upper_quantile_nonfollow_update_num_mode3, alpha=0.3, color = COLOR_NONFOLLOW3)
        std_fence_update_num_mode3 = np.std(my_conv(result_mode3[3]), axis=1)
        lower_quantile_fence_update_num_mode3 = np.quantile(my_conv(result_mode3[3]), LP, axis=1)
        upper_quantile_fence_update_num_mode3 = np.quantile(my_conv(result_mode3[3]), UP, axis=1)
        plt.errorbar([T], fence_update_num_mode3[-1], yerr=2*std_fence_update_num_mode3[-1]/np.sqrt(Runnum), fmt='o', capsize = 5, color = COLOR_FENCE3) #2 sigma
        plt.fill_between(range(1, len(Regs_mode2)+1), lower_quantile_fence_update_num_mode3, upper_quantile_fence_update_num_mode3, alpha=0.3, color = COLOR_FENCE3)
    #plt.plot(range(1, len(Regs_mode3)+1), follow_update_num_mode3 + nonfollow_update_num_mode3 + fence_update_num_mode3, label = "ACC(t)", color = COLOR_TOTAL3, linestyle = LINESTYLE_TOTAL3)
    plt.plot(range(1, len(Regs_mode3)+1), follow_update_num_mode3, label = "Obey", color = COLOR_FOLLOW3, linestyle = LINESTYLE_FOLLOW3)
    plt.plot(range(1, len(Regs_mode3)+1), nonfollow_update_num_mode3, label = "Deviate", color = COLOR_NONFOLLOW3, linestyle = LINESTYLE_NONFOLLOW3)
    plt.plot(range(1, len(Regs_mode3)+1), fence_update_num_mode3, label = "OtF", color = COLOR_FENCE3, linestyle = LINESTYLE_FENCE3)
    plt.legend()
    plt.ylabel("Number of Updates")
    plt.xlabel("Round (t)")
    my_show()
    if save_img:
        fig.savefig("update_num3.pdf", dpi=fig.dpi, bbox_inches='tight')
        colab_save("update_num3.pdf")
    plt.clf()

    fig = plt.figure(figsize=Figsize)
    follow_update_mode2 = my_conv(np.sum(result_mode2[4], axis=1)/Runnum)
    nonfollow_update_mode2 = my_conv(np.sum(result_mode2[5], axis=1)/Runnum)
    total_update_mode2 = follow_update_mode2 + nonfollow_update_mode2
    if confidence_bound:
        lower_quantile_total_update_mode2 = np.quantile(my_conv(result_mode2[4]+result_mode2[5]), LP, axis=1)
        upper_quantile_total_update_mode2 = np.quantile(my_conv(result_mode2[4]+result_mode2[5]), UP, axis=1)
        std_total_update_mode2 = np.std(my_conv(result_mode2[4]+result_mode2[5]), axis=1)
        plt.errorbar([T], total_update_mode2[-1], yerr=2*std_total_update_mode2[-1]/np.sqrt(Runnum), fmt='o', capsize = 5, color = COLOR_TOTAL2) #2 sigma
        plt.fill_between(range(1, len(Regs_mode2)+1), lower_quantile_total_update_mode2, upper_quantile_total_update_mode2, alpha=0.3, color = COLOR_TOTAL2)
        lower_quantile_follow_update_mode2 = np.quantile(my_conv(result_mode2[4]), LP, axis=1)
        upper_quantile_follow_update_mode2 = np.quantile(my_conv(result_mode2[4]), UP, axis=1)
        std_follow_update_mode2 = np.std(my_conv(result_mode2[4]), axis=1)
        plt.errorbar([T], follow_update_mode2[-1], yerr=2*std_follow_update_mode2[-1]/np.sqrt(Runnum), fmt='o', capsize = 5, color = COLOR_FOLLOW2) #2 sigma
        plt.fill_between(range(1, len(Regs_mode2)+1), lower_quantile_follow_update_mode2, upper_quantile_follow_update_mode2, alpha=0.3, color = COLOR_FOLLOW2)
        std_nonfollow_update_mode2 = np.std(my_conv(result_mode2[5]), axis=1)
        lower_quantile_nonfollow_update_mode2 = np.quantile(my_conv(result_mode2[5]), LP, axis=1)
        upper_quantile_nonfollow_update_mode2 = np.quantile(my_conv(result_mode2[5]), UP, axis=1)
        plt.errorbar([T], nonfollow_update_mode2[-1], yerr=2*std_nonfollow_update_mode2[-1]/np.sqrt(Runnum), fmt='o', capsize = 5, color = COLOR_NONFOLLOW2) #2 sigma
        plt.fill_between(range(1, len(Regs_mode2)+1), lower_quantile_nonfollow_update_mode2, upper_quantile_nonfollow_update_mode2, alpha=0.3, color = COLOR_NONFOLLOW2)
    plt.plot(range(1, len(Regs_mode2)+1), follow_update_mode2 + nonfollow_update_mode2, label = "ACC(t)", color = COLOR_TOTAL2, linestyle = LINESTYLE_TOTAL2)
    plt.plot(range(1, len(Regs_mode2)+1), follow_update_mode2, label = "ACC(t)_Obey", color = COLOR_FOLLOW2, linestyle = LINESTYLE_FOLLOW2)
    plt.plot(range(1, len(Regs_mode2)+1), nonfollow_update_mode2, label = "ACC(t)_Deviate", color = COLOR_NONFOLLOW2, linestyle = LINESTYLE_NONFOLLOW2)
    plt.legend()
    plt.ylabel("Amount of update")
    plt.xlabel("Round (t)")
    my_show()
    if save_img:
        fig.savefig("update2.pdf", dpi=fig.dpi, bbox_inches='tight')
        colab_save("update2.pdf")
    plt.clf()

    fig = plt.figure(figsize=Figsize)
    follow_update_mode3 = my_conv(np.sum(result_mode3[4], axis=1)/Runnum)
    nonfollow_update_mode3 = my_conv(np.sum(result_mode3[5], axis=1)/Runnum)
    fence_update_mode3 = my_conv(np.sum(result_mode3[6], axis=1)/Runnum)
    total_update_mode3 = follow_update_mode3 + nonfollow_update_mode3 + fence_update_mode3
    if confidence_bound:
        #lower_quantile_total_update_mode3 = np.quantile(my_conv(result_mode3[4]+result_mode3[5]+result_mode3[6]), LP, axis=1)
        #upper_quantile_total_update_mode3 = np.quantile(my_conv(result_mode3[4]+result_mode3[5]+result_mode3[6]), UP, axis=1)
        #std_total_update_mode3 = np.std(my_conv(result_mode3[4]+result_mode3[5]+result_mode3[6]), axis=1)
        #plt.errorbar([T], total_update_mode3[-1], yerr=2*std_total_update_mode3[-1]/np.sqrt(Runnum), fmt='o', capsize = 5, color = COLOR_TOTAL3) #2 sigma
        #plt.fill_between(range(1, len(Regs_mode2)+1), lower_quantile_total_update_mode3, upper_quantile_total_update_mode3, alpha=0.3, color = COLOR_TOTAL3)
        lower_quantile_follow_update_mode3 = np.quantile(my_conv(result_mode3[4]), LP, axis=1)
        upper_quantile_follow_update_mode3 = np.quantile(my_conv(result_mode3[4]), UP, axis=1)
        std_follow_update_mode3 = np.std(my_conv(result_mode3[4]), axis=1)
        plt.errorbar([T], follow_update_mode3[-1], yerr=2*std_follow_update_mode3[-1]/np.sqrt(Runnum), fmt='o', capsize = 5, color = COLOR_FOLLOW3) #2 sigma
        plt.fill_between(range(1, len(Regs_mode2)+1), lower_quantile_follow_update_mode3, upper_quantile_follow_update_mode3, alpha=0.3, color = COLOR_FOLLOW3)
        std_nonfollow_update_mode3 = np.std(my_conv(result_mode3[5]), axis=1)
        lower_quantile_nonfollow_update_mode3 = np.quantile(my_conv(result_mode3[5]), LP, axis=1)
        upper_quantile_nonfollow_update_mode3 = np.quantile(my_conv(result_mode3[5]), UP, axis=1)
        plt.errorbar([T], nonfollow_update_mode3[-1], yerr=2*std_nonfollow_update_mode3[-1]/np.sqrt(Runnum), fmt='o', capsize = 5, color = COLOR_NONFOLLOW3) #2 sigma
        plt.fill_between(range(1, len(Regs_mode2)+1), lower_quantile_nonfollow_update_mode3, upper_quantile_nonfollow_update_mode3, alpha=0.3, color = COLOR_NONFOLLOW3)
        std_fence_update_mode3 = np.std(my_conv(result_mode3[6]), axis=1)
        lower_quantile_fence_update_mode3 = np.quantile(my_conv(result_mode3[6]), LP, axis=1)
        upper_quantile_fence_update_mode3 = np.quantile(my_conv(result_mode3[6]), UP, axis=1)
        plt.errorbar([T], fence_update_mode3[-1], yerr=2*std_fence_update_mode3[-1]/np.sqrt(Runnum), fmt='o', capsize = 5, color = COLOR_FENCE3) #2 sigma
        plt.fill_between(range(1, len(Regs_mode2)+1), lower_quantile_fence_update_mode3, upper_quantile_fence_update_mode3, alpha=0.3, color = COLOR_FENCE3)
    #plt.plot(range(1, len(Regs_mode3)+1), follow_update_mode3 + nonfollow_update_mode3+ fence_update_mode3, label = "ACC(t)", color = COLOR_TOTAL3, linestyle = LINESTYLE_TOTAL3)
    plt.plot(range(1, len(Regs_mode3)+1), follow_update_mode3, label = "ACC(t)_Obey", color = COLOR_FOLLOW3, linestyle = LINESTYLE_FOLLOW3)
    plt.plot(range(1, len(Regs_mode3)+1), nonfollow_update_mode3, label = "ACC(t)_Deviate", color = COLOR_NONFOLLOW3, linestyle = LINESTYLE_NONFOLLOW3)
    plt.plot(range(1, len(Regs_mode3)+1), fence_update_mode3, label = "ACC(t)_OtF", color = COLOR_FENCE3, linestyle = LINESTYLE_FENCE3)
    plt.legend()
    plt.ylabel("Amount of update")
    plt.xlabel("Round (t)")
    my_show()
    if save_img:
        fig.savefig("update3.pdf", dpi=fig.dpi, bbox_inches='tight')
        colab_save("update3.pdf")
    plt.clf()

# plot (single round)
def my_plot_single(results):
    ls, us, Reg1, Shrink1, Reg2, Shrink2, Reg2myopic, Shrink2myopic, \
        Reg2explore, Shrink2explore, Reg3, Shrink3 = results
    
    us_diff = us - ls
    fig = plt.figure(figsize=Figsize)
    plt.plot(us_diff, Reg2, label = "Binary (straight)", color = COLOR_MODE2, linestyle = LINESTYLE_MODE2, marker='x')
    plt.plot(us_diff, Reg2myopic, label = "Binary (myopic)", color = "tab:orange", linestyle = LINESTYLE_MODE2MYOPIC, marker='x')
    #plt.plot(us_diff, Reg2explore, label = "Binary (explore)", color = "green", linestyle = LINESTYLE_MODE2, marker='x')
    plt.plot(us_diff, Reg3, label = "Ternary", color = COLOR_MODE3, linestyle = LINESTYLE_MODE3, marker='x')
    plt.ylabel("Expected per-round regret")
    plt.xlabel("Width of confidence interval")
    plt.legend()
    my_show()
    if save_img:
        fig.savefig("single_short.pdf", dpi=fig.dpi, bbox_inches='tight')
        colab_save("single_short.pdf")
    plt.clf()

    # x100 for percentage
    fig = plt.figure(figsize=Figsize)
    plt.plot(us_diff, 100*Shrink2/us_diff, label = "Binary (straight)", color = COLOR_MODE2, linestyle = LINESTYLE_MODE2, marker='x')
    plt.plot(us_diff, 100*Shrink2myopic/us_diff, label = "Binary (myopic)", color = "tab:orange", linestyle = LINESTYLE_MODE2MYOPIC, marker='x')
    #plt.plot(us_diff, Shrink2explore/us_diff, label = "Binary (explore)", color = "green", linestyle = LINESTYLE_MODE2, marker='x')
    plt.plot(us_diff, 100*Shrink3/us_diff, label = "Ternary", color = COLOR_MODE3, linestyle = LINESTYLE_MODE3, marker='x')
    plt.ylabel("Expected percentage shrink (%)")
    plt.xlabel("Width of confidence interval")
    plt.legend()
    my_show()
    if save_img:
        fig.savefig("single_long.pdf", dpi=fig.dpi, bbox_inches='tight')
        colab_save("single_long.pdf")

    fig = plt.figure(figsize=Figsize)
    #plt.plot(us_diff, Reg1, label = "Unitary", color = "tab:olive", linestyle = LINESTYLE_MODE1, marker='x')
    plt.plot(us_diff, Reg2, label = "Binary (straight)", color = COLOR_MODE2, linestyle = LINESTYLE_MODE2, marker='x')
    plt.plot(us_diff, Reg2myopic, label = "Binary (myopic)", color = "tab:orange", linestyle = LINESTYLE_MODE2MYOPIC, marker='x')
    plt.plot(us_diff, Reg2explore, label = "Binary (explore)", color = "green", linestyle = LINESTYLE_MODE2EVE, marker='x')
    plt.plot(us_diff, Reg3, label = "Ternary", color = COLOR_MODE3, linestyle = LINESTYLE_MODE3, marker='x')
    plt.ylabel("Expected per-round regret")
    plt.xlabel("Width of confidence interval")
    plt.legend()
    my_show()
    if save_img:
        fig.savefig("single_short_inclexplore.pdf", dpi=fig.dpi, bbox_inches='tight')
        colab_save("single_short_inclexplore.pdf")
    plt.clf()

    print(f"wt = {us_diff}")
    print(f"Straight reg= {Reg2}")
    print(f"Myopic reg= {Reg2myopic}")
    print(f"Explore reg= {Reg2explore}")
    print(f"Ternary reg= {Reg3}")

    fig = plt.figure(figsize=Figsize)
    plt.plot(us_diff, 100*Shrink2/us_diff, label = "Binary (straight)", color = COLOR_MODE2, linestyle = LINESTYLE_MODE2, marker='x')
    plt.plot(us_diff, 100*Shrink2myopic/us_diff, label = "Binary (myopic)", color = "tab:orange", linestyle = LINESTYLE_MODE2MYOPIC, marker='x')
    plt.plot(us_diff, 100*Shrink2explore/us_diff, label = "Binary (explore)", color = "green", linestyle = LINESTYLE_MODE2EVE, marker='x')
    plt.plot(us_diff, 100*Shrink3/us_diff, label = "Ternary", color = COLOR_MODE3, linestyle = LINESTYLE_MODE3, marker='x')
    plt.ylabel("Expected percentage shrink (%)")
    plt.xlabel("Width of confidence interval")
    plt.legend()
    my_show()
    if save_img:
        fig.savefig("single_long_inclexplore.pdf", dpi=fig.dpi, bbox_inches='tight')
        colab_save("single_long_inclexplore.pdf")

    print(f"wt = {us_diff}")
    print(f"Straight shrink= {100*Shrink2/us_diff}")
    print(f"Myopic shrink= {100*Shrink2myopic/us_diff}")
    print(f"Explore shrink= {100*Shrink2explore/us_diff}")
    print(f"Ternary shrink= {100*Shrink3/us_diff}")


if True:
    np.random.seed(1) #fix random seed
    if True: # simulation (note: run=5000 takes ~12 hours on 20core desktop)
        result_mode2 = main(mode = "mode2")
        result_mode2myopic = main(mode = "mode2_myopic")
        result_mode2explore = main(mode = "mode2_explore")
        result_mode3 = main(mode = "mode3")
        # save data
        all = (result_mode2, result_mode2myopic, result_mode2explore, result_mode3)
        pickle.dump( all, open( "all.pickle", "wb" ) )
    else: # use previous simulation results
        file = open("all.pickle",'rb')
        (result_mode2, result_mode2myopic, result_mode2explore, result_mode3) = pickle.load(file)

    my_plot(result_mode2, result_mode2myopic, result_mode2explore, result_mode3)

if True:
    # single
    np.random.seed(1) #fix random seed
    if True: # simulation
        results_single = compare_singleround()
        # save data
        all = (results_single)
        pickle.dump( all, open( "results_single.pickle", "wb" ) )
    else: # use previous simulation results
        file = open("results_single.pickle",'rb')
        results_single = pickle.load(file)

    my_plot_single(results_single)