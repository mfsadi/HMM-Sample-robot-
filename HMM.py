import re
import numpy as np
'''
#-------------------------------------------------------------------------------------------------#
                                In the name of God
This tiny program tries to build an HMM from some input file which is based on a robot movements.
The robot moves is in a 4*4 area which some of the blocks are holes. If it decides to go to holes
or move beyond the confines of the area it remains in it's last place. There are 400 sequences of
continuous walking, 200 for training and 200 for test. Each sequence contains steps like 3:3 r  ,
which 3:3 show the dimensions of the state and 'r' show the observed color. The goal is  to train
HMM to predict the best sequence of states to observe a given sequence of observations.
                        ____ ____ ____ ____
                       |hole|_r__|_y__|_b__|
                       |_g__|_b__|_r__|hole|
                       |_r__|hole|_g__|_y__|
                       |hole|_g__|_y__|_b__|

 'r' = red, 'b' = blue, 'g' = green and 'y' = yellow
 ------M.F.Saadi, Zanjan University, Computer Department, 11.19.2016, mfsadi@znu.ac.ir----
 * Viterbi code added on 11/25/2016, output goes to out.data, input and output files must be in 
 the current directory beside the program.Input files are: robot_train.data and robot_test.data.
#-------------------------------------------------------------------------------------------------#
'''
# Declarations
# States
s0 = '1:1'
s1 = '2:1'
s2 = '3:1'
s3 = '4:1'
s4 = '1:2'
s5 = '2:2'
s6 = '3:2'
s7 = '4:2'
s8 = '1:3'
s9 = '2:3'
s10 = '3:3'
s11 = '4:3'
s12 = '1:4'
s13 = '2:4'
s14 = '3:4'
s15 = '4:4'
test = 4
# observations
o1 = 'r'
o2 = 'b'
o3 = 'g'
o4 = 'y'
# viterbi variables
v_o = []
v_n = []
v_f = []
vmax = 0
v_flag = 1
v_temp = []
temp16 = 0
counter = 0
# Probabilities
# Starting State
p_s = []
# Conditional probabilities
p_tr = []
temp = 'null'
oindex = 0
p_start = []
flag = 1
# initializations

for k in range(0, 16):  # Filling probabilities with zero
    p_s.append(0)
    v_o.append(0)
    v_n.append(0)
    p_start.append(0)
    v_temp.append(0)
counter = 0
p_o = []
new = []
for i in range(0, 4):
    for j in range(0, 16):
        new.append(0)
    p_o.append(new)
    new = []
for i in range(0, 16):
    for j in range(0, 16):
        new.append(0)
    p_tr.append(new)
    new = []
# HMM Building
print("Starting...")
with open('robot_train.data','r') as f:  # Opening Training Data File
        for line in f:
            if str(line.splitlines()) == "['..']":  # Delimiter which shows End Of Training Data
                break
            elif str(line.splitlines()) == "['.']":   # Delimiter which shows End Of each Sequence of robot walks
                counter += 1
                if counter % 20 == 0:
                    x = float((counter/200)*100)
                    print('Working .... (%d percent)' % x)
                flag = 1  # Shows if we are at the end of sequence (.)
                temp = 'null'  # To hold previous String of state and observation
                oindex = 0  # To hold previous index for updating probabilities
            else:
                seq = str(line.splitlines())  # Splitting input line by line
                state = seq[2:5]  # Holding just State (ex, 2:3)
                obs = seq[6]  # Holding an observation (ex, 'r' for red)
                x = int(seq[2])  # First dimension of area
                y = int(seq[4])  # Second dimension of area
                index = ((y-1)*4)+(x-1)  # Index to map an state dimensions to an array of 16 elements
# Calculating transitional probabilities
                if flag == 1:
                    p_start[index] += 1  # Updating probability of Starting states
                    flag = 0
                if temp == 'null':
                    temp = seq  # Holding last state
                    oindex = index
                else:
                    p_tr[oindex][index] += 1  # Updating Conditional probabilities (P(O|S))
                    temp = seq
                    oindex = index
# Calculating conditional probabilities
# for observation : red
                if obs=='r':
                    p_o[0][index] += 1
                    p_s[index] += 1  # Counting total number of visiting a specific state
# for observation : blue
                if obs == 'b':
                    p_o[1][index] += 1
                    p_s[index] += 1
# for observation : green
                if obs == 'g':
                    p_o[2][index] += 1
                    p_s[index] += 1
# for observation : yellow
                if obs == 'y':
                    p_o[3][index] += 1
                    p_s[index] += 1
f.close()  # Closing the file

def cond_prob():  # Calculating Conditional probabilities
    for i in range(0, 4):
        for j in range(0, 16):
            p_o[i][j] = float((p_o[i][j]+1)/(p_s[j]+4))  # Laplacian Smoothing


def tr_prob():  # Calculating Transitional probabilities
    for i in range(0, 16):
        for j in range(0, 16):
            p_tr[i][j] = float((p_tr[i][j]+1)/(p_s[i]+16))  # Laplacian Smoothing


def start_prob():  # Calculating starting probabilities
    for i in range(0, 16):
        p_start[i] = float((p_start[i]+1)/(200+16))  # Laplacian Smoothing

cond_prob()
tr_prob()
start_prob()

# Implementing Viterbi algorithm to predict hidden states based on observations in test file


def match_obs (s):  # This function matches color observations to some numbers
    ind=0
    if s == 'r':
        ind = 0
    elif s == 'b':
        ind = 1
    elif s == 'g':
        ind = 2
    elif s == 'y':
        ind = 3
    return (ind)


def unmap(s):   # This function convert states in matrix indices (ex. 10) to predefined variables (ex,3:3)
    for i in range(0, len(s)):
        temp22 = s[i]
        s[i] = globals()['s' + str(temp22)]
    return s


def viterbi():  # Doing Viterbi on test data
    v_flag = 1
    with open('robot_test.data','r') as f:  # Opening test Data File
        for line in f:
            if str(line.splitlines()) == "['..']":  # Delimiter which shows End Of Test Data
                break
            elif str(line.splitlines()) == "['.']":   # Delimiter which shows End Of each Sequence of robot walks
                v_flag = 1
            else:
                vseq = str(line.splitlines())  # spiliting line
                vobs = vseq [6]  # Holds the observation
                viobs = match_obs(vobs)
                if (v_flag == 1):   # Starting point of sequence
                    for i in range(0, 16):
                        v_o [i] = p_start[i] * p_o[viobs][i]  # First step
                    vmax = max (v_o)    # Maximizing
                    temp3 = v_o.index(vmax)
                    v_f.append(temp3)  # v_f holds entire sequence of detected hidden states
                    v_flag = 0
                else:  # Forward recursion
                    for i in range(0,16):
                        v_temp[i] = 0
                    for j in range(0, 16):
                        for k in range(0, 16):
                            v_temp[k] = 0
                            v_temp[k] = v_temp[k] + (v_o[k] * p_o[viobs][j] * p_tr[k][j])
                    # Multiplication of observation probability and transitional probability and this value in previous state
                        v_n[j] = max (v_temp)  # Maximizing, This gives just one out of sixteen state probability so far.
                    vmax = max (v_n)  # This is the Maximum probability state for this observation
                    temp2 = v_n.index(vmax)
                    v_f.append(temp2)  # Append result to list
                    for k in range(0, 16):
                        v_o[k] = v_n[k]  # Holding last step results for recursion
    f.close()  # Closing the file
viterbi()
# Results
v_f = unmap(v_f)  # Convert state x into Matrix index x:y
with open('out.data', 'w') as f:  # Writing Results to
    for w in range(0,len(v_f)):
        if w % 200 == 0:
            if w > 0:
                f.write(str("."+"\n"))  # End of sequence
        f.write(str(v_f[w])+"\n")
    f.write("..")  # End of File
f.close()
print("Work is Done! out.data was created.")