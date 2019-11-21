# -*- coding: utf-8 -*-
"""
Created on Wed Nov 20 20:46:44 2019

@author: 9730j
"""

import numpy as np
import probabilistic_serial_mechanism as ps
from constrained_birkhoff_von_neumann import constrained_birkhoff_von_neumann_decomposition as bvn
stuA=np.zeros(500)
stuB=np.zeros(500)
stu=np.concatenate((stuA,stuB))
objAp=np.zeros(20)
objAup=np.zeros(20)
objA=np.concatenate((objAp,objAup))
objBp=np.zeros(20)
objBup=np.zeros(20)
objB=np.concatenate((objBp,objBup))
objEp=np.zeros(20)
objEup=np.zeros(20)
objE=np.concatenate((objEp,objEup))
obj=np.concatenate((objA,objB,objE))
stu_num=len(stu)
stuA_num=len(stuA)
stuB_num=len(stuB)
obj_num=len(obj)
objAp_num=len(objAp)
objAup_num=len(objAup)
objA_num=len(objA)
objBp_num=len(objBp)
objBup_num=len(objBup)
objB_num=len(objB)
objEp_num=len(objEp)
objEup_num=len(objEup)
objE_num=len(objE)
#utility matrix
E=np.random.rand(stu_num,obj_num)
TAP=np.random.rand(stu_num,objAp_num)
TBP=np.random.rand(stu_num,objBp_num)
TEP=np.random.rand(stu_num,objEp_num)
TAUP=np.zeros((stu_num,objAup_num))
TBUP=np.zeros((stu_num,objBup_num))
TEUP=np.zeros((stu_num,objEup_num))
T=np.concatenate((TAP,TAUP,TBP,TBUP,TEP,TEUP),axis=1)
print(T)
VAA=np.vstack((np.random.rand(stuA_num,objA_num),np.zeros((stuB_num,objA_num))))
VBB=np.vstack((np.zeros((stuA_num,objB_num)),2*np.random.rand(stuB_num,objB_num)))
V=np.hstack((VAA,VBB,np.zeros((stu_num,objE_num))))
utility=E+T+V
import random
for stu_id in range(stu_num):
    for obj_id in range(obj_num):
        indic=np.array([1 if stu_id>=500 and obj_id<40 else 0])
        utility[stu_id][obj_id]+=indic*random.random()
print(utility)

#preference list(truthful)
pref=dict()
R=dict()
for stu_id in range(stu_num):
        for obj_id in range(obj_num):
            pref[obj_id] = np.where(np.sort(utility[stu_id])[::-1] == utility[stu_id][obj_id])[0][0]
        new_pref=dict()
        for k,v in pref.items():
            new_pref[v]=k
        snew_pref=sorted(new_pref.items())
        preflist=list()
        for k,v in snew_pref:
            preflist.append(v)
        R[stu_id]=preflist
print(R)
#behavior matrix
behave=np.zeros((stu_num,obj_num))
for stu_id in range(stu_num):
    for obj_id in range(obj_num):
        behave[stu_id][obj_id]=random.uniform(utility[stu_id][obj_id]-1/2,utility[stu_id][obj_id]+1/2)
print(behave)
#reported preference list
report=dict()
reportedR=dict()
for stu_id in range(stu_num):
        for obj_id in range(obj_num):
            report[obj_id] = np.where(np.sort(behave[stu_id])[::-1] == behave[stu_id][obj_id])[0][0]
        new_report=dict()
        for k,v in report.items():
            new_report[v]=k
        snew_report=sorted(new_report.items())
        reportlist=list()
        for k,v in snew_report:
            reportlist.append(v)
        reportedR[stu_id]=reportlist
print(reportedR)

#capacity
cap=list()
for i in range(obj_num):
    cap.append(50)
print(cap)

#Criteria
def averf(X):
    stu_num=len(X)
    allo_num=len(X[0])
    rr=list()
    for stu_id in range(stu_num):
        X1=X[stu_id]
        r=np.sum(X1)/allo_num
        rr.append(r)
    aver2=sum(rr)/len(rr)
    return aver2

def fprobf(X):
    fprob2=np.sum(X[:,0])/len(X[:,0])
    return fprob2
    
    
#strategic behavior case, quota_list is list, quota is scalar
import matplotlib.pyplot as plt

def simulation(repeat,quota_list):
    average_ranks=list()
    first_probs=list()
    for i in range(len(quota_list)):
        quota=quota_list[i]
        constraint_structure=dict()
        for stu_id in range(stu_num):
            c=list()
            for obj_id in range(obj_num):
                c.append((stu_id,obj_id))
            fc=frozenset(c)
            constraint_structure[fc]=(6,6)
        for obj_id in range(obj_num):
            c=list()
            for stu_id in range(stu_num):
                 c.append((stu_id,obj_id))
            fc=frozenset(c)
            constraint_structure[fc]=(0,50)
        for obj_id in range(0,objA_num):
            c=list()
            for stu_id in range(stu_num-stuA_num,stu_num):
                c.append((stu_id,obj_id))
            fc=frozenset(c)
            constraint_structure[fc]=(0,quota)
        for obj_id in range(objA_num,objA_num+objB_num):
            c=list()
            for stu_id in range(0,stuA_num):
                c.append((stu_id,obj_id))
            fc=frozenset(c)
            constraint_structure[fc]=(0,quota)
        avers=list()
        fprobs=list()
        for i in range(repeat):
            allodict,X=ps.modified_probabilistic_serial_mechanism(reportedR,cap)
            postallo=bvn(X,constraint_structure)
            aver=averf(X)
            fprob=fprobf(X)
            avers.append(aver)
            fprobs.append(fprob)
        average_rank=sum(avers)/len(avers)
        first_prob=sum(fprobs)/len(fprobs)
        average_ranks.append(average_rank)
        first_probs.append(first_prob)
    return average_ranks, first_probs
    plt.plot(quota_list,average_ranks,'b',label='average rank')
    plt.plot(quota_list,first_probs,'r',label='getting first course probability')
    plt.xlabel('major quota')
    plt.ylabel('rank or probability')
    plt.title('modified probabilistic serial mechanism')
    plt.legend(loc='upper right')
    plt.show()
    
simulation(repeat=50,quota_list=[5,10,15,20,25,30])
    