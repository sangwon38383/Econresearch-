import numpy as np
import probabilistic_serial_mechanism as ps
import random
stuA=np.zeros(50)
stuB=np.zeros(50)
stu=np.concatenate((stuA,stuB))
objAp=np.zeros(5)
objAup=np.zeros(5)
objA=np.concatenate((objAp,objAup))
objBp=np.zeros(5)
objBup=np.zeros(5)
objB=np.concatenate((objBp,objBup))
objEp=np.zeros(10)
objEup=np.zeros(10)
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
def utilitymat():
    E=np.random.rand(stu_num,obj_num)
    TAP=np.random.rand(stu_num,objAp_num)
    TBP=np.random.rand(stu_num,objBp_num)
    TEP=np.random.rand(stu_num,objEp_num)
    TAUP=np.zeros((stu_num,objAup_num))
    TBUP=np.zeros((stu_num,objBup_num))
    TEUP=np.zeros((stu_num,objEup_num))
    T=np.concatenate((TAP,TAUP,TBP,TBUP,TEP,TEUP),axis=1)
    VAA=np.vstack((np.random.rand(stuA_num,objA_num),np.zeros((stuB_num,objA_num))))
    VBB=np.vstack((np.zeros((stuA_num,objB_num)),2*np.random.rand(stuB_num,objB_num)))
    V=np.hstack((VAA,VBB,np.zeros((stu_num,objE_num))))
    utility=E+T+V
    import random
    for stu_id in range(stu_num):
        for obj_id in range(obj_num):
            indic=np.array([1 if stu_id>=stuA_num and obj_id<objA_num else 0])
            utility[stu_id][obj_id]+=indic*random.random()
    return utility

#preference list(truthful)
def preferlist():
    pref=dict()
    R=dict()
    utility=utilitymat()
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
    return R

#behavior matrix
def behaviormat():
    behave=np.zeros((stu_num,obj_num))
    utility=utilitymat()
    for stu_id in range(stu_num):
        for obj_id in range(obj_num):
            behave[stu_id][obj_id]=random.uniform(utility[stu_id][obj_id]-1/2,utility[stu_id][obj_id]+1/2)
    return behave

#reported preference list
def reppreferlist():
    report=dict()
    reportedR=dict()
    behave=behaviormat()
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
    return reportedR

#capacity
cap=list()
for i in range(obj_num):
    cap.append(7)
print(cap)

#Criteria for average rank
def averf(X):
    stu_num=len(X)
    rr=list()
    R=preferlist()
    for stu_id in range(stu_num):
        X1=X[stu_id]
        X2=list()
        for i in range(obj_num):
            j=R[stu_id].index(i)
            j+=1
            X2.append(j)
        X22=np.array(X2)
        r=np.dot(X1,X22)/np.sum(X1)
        rr.append(r)
    aver2=sum(rr)/len(rr)
    return aver2
#probability to get the most preferred course
def fprobf(X):
    R=preferlist()
    e=list()
    for stu_id,v in R.items():
        e.append(X[stu_id][v[0]])
    fprob2=sum(e)/len(e)
    return fprob2
#envyness
def envyf(X):
    envy=list()
    R=preferlist()
    for stu_id in range(stu_num):
        envyc=0
        Y2=list()
        Y3=X[stu_id]
        for i in range(obj_num):
            j=R[stu_id].index(i)
            j+=1
            Y2.append(j)
        Y22=np.array(Y2)
        for i in range(stu_num):
            Y1=X[i]
            y=np.dot(Y1,Y22)/np.sum(Y1)
            if y<np.dot(Y3,Y22)/np.sum(Y3):
                envyc+=1
        envy.append(envyc)
    envy_per_stu=sum(envy)/stu_num
    return envy_per_stu        
#proportion of popular major student(A) in each general education course
def fgepro(X):
    L6=list()
    for i in range(objA_num+objB_num, obj_num):
        L=list()
        for stu_id in range(stuA_num):
            L.append(X[stu_id][i])
        L2=sum(L)
        L3=list()
        for stu_id in range(stuA_num,stu_num):
            L3.append(X[stu_id][i])
        L4=sum(L3)
        if L2+L4!=0:    
            L5=L2/(L2+L4)
        else:
            L5=1/2
        L6.append(L5)
    return L6
#proportion of popular major student(A) in each their major's popular course(Apop)
def fAppro(X):
    L7=list()
    for i in range(objAp_num):
        L8=list()
        for stu_id in range(stuA_num):
            L8.append(X[stu_id][i])
        L9=sum(L8)
        L10=list()
        for stu_id in range(stuA_num,stu_num):
            L10.append(X[stu_id][i])
        L11=sum(L10)
        if L9+L11!=0:
            L12=L9/(L9+L11)
        else:
            L12=1/2
        L7.append(L12)
    return L7

#strategic behavior case, quota_list is list, quota is scalar
import matplotlib.pyplot as plt

def simulation(repeat,quota_list):
    average_ranks=list()
    first_probs=list()
    envy_counts=list()
    K4=list()
    KK4=list()
    for i in range(len(quota_list)):
        quota=quota_list[i]
        avers=list()
        fprobs=list()
        envycounts=list()
        LL1s=list()
        LL2s=list()
        for i in range(repeat):
            reportedR=reppreferlist()
            allodict,X=ps.modified_probabilistic_serial_mechanism(reportedR, cap, quota, obj_num, objA_num, objB_num, stu_num, stuA_num, stuB_num)
            aver=averf(X)
            fprob=fprobf(X)
            envycount=envyf(X)
            LL1=fgepro(X)
            LL2=fAppro(X)
            avers.append(aver)
            fprobs.append(fprob)
            envycounts.append(envycount)
            LL1s.append(LL1)
            LL2s.append(LL2)
        average_rank=sum(avers)/len(avers)
        first_prob=sum(fprobs)/len(fprobs)
        envy_count=sum(envycounts)/len(envycounts)
        K3=list()
        for k in range(objE_num):
            K1=list()
            for j in range(repeat):
                K1.append(LL1s[j][k])
            K2=sum(K1)/len(K1)
            K3.append(K2)
        K4.append(K3)
        KK3=list()
        for k in range(objAp_num):
            KK1=list()
            for j in range(repeat):
                KK1.append(LL2s[j][k])
            KK2=sum(KK1)/len(KK1)
            KK3.append(KK2)
        KK4.append(KK3)
        average_ranks.append(average_rank)
        first_probs.append(first_prob)
        envy_counts.append(envy_count)
    return average_ranks, first_probs, envy_counts, K4, KK4

qqq=[50,30,25,20,15,10,5]    
rrr=1000
average_ranks,first_probs,envy_counts, proportion_ge, proportion_Ap=simulation(rrr,qqq)
plt.plot(['No','30','25','20','15','10','5'],average_ranks,'b', marker='o')
plt.xlabel('major quota')
plt.ylabel('average rank')
plt.title('Efficieny-average welfare')
fig = plt.gcf()
fig.savefig('Efficieny-average welfare.png')
plt.show()
fig.savefig('Efficieny-average welfare.png')
plt.plot(['No','30','25','20','15','10','5'],first_probs,'r', marker='o')
plt.xlabel('major quota')
plt.ylabel('getting first course probability')
plt.title('Efficieny-representative welfare')
fig = plt.gcf()
fig.savefig('Efficieny-representative welfare.png')
plt.show()
fig.savefig('Efficieny-representative welfare.png')
plt.plot(['No','30','25','20','15','10','5'],envy_counts,'g', marker='o')
plt.xlabel('major quota')
plt.ylabel('envyness per student')
plt.title('Fairness-average envyness per student')
fig = plt.gcf()
fig.savefig('Fairness.png')
plt.show()
fig.savefig('Fairness.png')

ppp1=[]
for i in range(objA_num+objB_num,obj_num):
  ppp1.append(i)
plt.plot(ppp1, proportion_ge[0], marker='o',label='No quota')
plt.plot(ppp1, proportion_ge[1], marker='o',label='quota 30')
plt.plot(ppp1, proportion_ge[2], marker='o',label='quota 25')
plt.plot(ppp1, proportion_ge[3], marker='o',label='quota 20')
plt.plot(ppp1, proportion_ge[4], marker='o',label='quota 15')
plt.plot(ppp1, proportion_ge[5], marker='o',label='quota 10')
plt.plot(ppp1, proportion_ge[6], marker='o',label='quota 5')
plt.xlabel('general education courses')
plt.ylabel('proportion of popular major student(A)')
plt.title('modified probabilistic serial mechanism effectiveness in general education courses')
plt.legend(loc='upper right')
fig = plt.gcf()
fig.savefig('general edu success proportion.png')
plt.show()
fig.savefig('general edu success proportion.png')

ppp2=[]
for i in range(0,objAp_num):
  ppp2.append(i)
plt.plot(ppp2, proportion_Ap[0], marker='o', label='No quota')
plt.plot(ppp2, proportion_Ap[1], marker='o', label='quota 30')
plt.plot(ppp2, proportion_Ap[2], marker='o', label='quota 25')
plt.plot(ppp2, proportion_Ap[3], marker='o', label='quota 20')
plt.plot(ppp2, proportion_Ap[4], marker='o', label='quota 15')
plt.plot(ppp2, proportion_Ap[5], marker='o', label='quota 10')
plt.plot(ppp2, proportion_Ap[6], marker='o', label='quota 5')
plt.xlabel('popular courses in popular major')
plt.ylabel('proportion of popular major student(A)')
plt.title('modified probabilistic serial mechanism effectiveness in double-popular courses')
plt.legend(loc='upper right')
fig = plt.gcf()
fig.savefig('double popular course success proportion.png')
plt.show()
fig.savefig('double popular course success proportion.png')

