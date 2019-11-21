# -*- coding: utf-8 -*-
"""
Created on Wed Nov 20 20:57:45 2019

@author: 9730j
"""

import numpy as np
import copy

TOLERANCE = 1.0e-5

def probabilistic_serial_mechanism(R, m):
  # define an empty dictionary to store the solution, and copies of R and m
  P={}
  Q = copy.deepcopy(R)
  t = copy.deepcopy(m)
  # give the empty dictionary P the structure of the solution
  for key, value in Q.items():
    P[key]= [0]*len(value)
  # eat probability mass while it remains
  while any(Q[key] != [] for key,values in Q.items()):
    # if an object has no remaining probability mass, remove it from the rank order lists
    for key,value in Q.items():
      Q[key] = [i for i in value if t[i] > TOLERANCE and P[key][i] < 1-TOLERANCE]
    # define a zero vector whose dimension equals the number of objects
    y = np.zeros(len(t))
    # define a vector of ones, one for each agent
    x = np.ones(len(Q.items()))
    # count how many agents rank each object first (under updated rank order lists)
    # for each agent's preferred object (under updated rank order lists), 
    # record in x how much more of that object's probability mass they can consume
    for key,value in Q.items():
      if value != []:
        y[value[0]] += 1
        x[key] = 1 - P[key][value[0]]
    # define a vector recording time taken until each object's probability mass is depleted without intervention
    z = [max(i,0.000001)/max(j,0.0000001) for i,j in zip(t,y)]
    # update probability masses m and record probability consumed in P
    for key,value in Q.items():
      if value != []:
        t[value[0]] -= min(min(z), min(x))
        P[key][value[0]] += min(min(z), min(x))
  # if all probaility masses are nil, the process is done—return the solution
  else:
    return P,np.array([value for key, value in P.items()])






TOLERANCE = 1.0e-5

def modified_probabilistic_serial_mechanism(R, m, quota, obj_num, objA_num, objB_num, stu_num, stuA_num, stuB_num):
  # define an empty dictionary to store the solution, and copies of R and m
  P={}
  Q = copy.deepcopy(R)
  t = copy.deepcopy(m)
  q=[quota]*obj_num
  # give the empty dictionary P the structure of the solution
  for key, value in Q.items():
    P[key]= [0]*len(value)
  # eat probability mass while it remains
  while any(Q[key] != [] for key,values in Q.items()):
    # if an object has no remaining probability mass, remove it from the rank order lists
    for key,value in Q.items():
        if 0<=key<stuA_num:
            Q[key] = [i for i in value if (objA_num<=i<objA_num+objB_num and q[i] > TOLERANCE and t[i] > TOLERANCE and P[key][i] < 1-TOLERANCE and sum(P[key])<6-TOLERANCE) or (0<=i<objA_num and t[i] > TOLERANCE and P[key][i] < 1-TOLERANCE and sum(P[key])<6-TOLERANCE)]
        elif stuA_num<=key<stu_num:
            Q[key] = [i for i in value if (0<=i<objA_num and q[i] > TOLERANCE and t[i] > TOLERANCE and P[key][i] < 1-TOLERANCE and sum(P[key])<6-TOLERANCE) or (objA_num<=i<objA_num+objB_num and t[i] > TOLERANCE and P[key][i] < 1-TOLERANCE and sum(P[key])<6-TOLERANCE)]
    # if an object has no remaining probability mass, remove it from the rank order lists
    # define a zero vector whose dimension equals the number of objects
    y = np.zeros(len(t))
    # define a vector of ones, one for each agent
    x = np.ones(len(Q.items()))
    yy=np.zeros(len(q))
    xx=np.ones(len(Q.items()))*6
    # count how many agents rank each object first (under updated rank order lists)
    # for each agent's preferred object (under updated rank order lists), 
    # record in x how much more of that object's probability mass they can consume
    for key,value in Q.items():
      if value != []:
        y[value[0]] += 1
        x[key] = 1 - P[key][value[0]]
        xx[key]=6-sum(P[key])
        if 0<=value[0]<objA_num and stuA_num<=key<stu_num:
            yy[value[0]]+=1
        elif objA_num<=value[0]<objA_num+objB_num and 0<=key<stuA_num:
            yy[value[0]]+=1
    # define a vector recording time taken until each object's probability mass is depleted without intervention
    z = [max(i,0.000001)/max(j,0.0000001) for i,j in zip(t,y)]
    zz=[max(i,0.000001)/max(j,0.0000001) for i,j in zip(q,yy)]
    # update probability masses m and record probability consumed in P
    for key,value in Q.items():
      if value != []:
        t[value[0]] -= min(min(z), min(x), min(xx), min(zz))
        P[key][value[0]] += min(min(z), min(x), min(xx), min(zz))
        if 0<=value[0]<objA_num and stuA_num<=key<stu_num:
            q[value[0]]-=min(min(z), min(x), min(xx), min(zz))
        elif objA_num<=value[0]<objA_num+objB_num and 0<=key<stuA_num:
            q[value[0]]-=min(min(z), min(x), min(xx), min(zz))
  # if all probaility masses are nil, the process is done—return the solution
  else:
    return P, np.array([value for key, value in P.items()])