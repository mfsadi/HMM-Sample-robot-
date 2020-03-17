# HMM-Sample-robot-
                                      In the name of God
This tiny program tries to build an HMM from some input file which is based on a robot movements.
The robot moves is in a 4*4 area which some of the blocks are holes. If it decides to go to holes
or move beyond the confines of the area it remains in it's last place. There are 400 sequences of
continuous walking, 200 for training and 200 for test. Each sequence contains steps like 3:3 r  ,
which 3:3 show the dimensions of the state and 'r' show the observed color. The goal is  to train
HMM to predict the best sequence of states to observe a given sequence of observations.
                        ____   ____   ____   ____
                        
                       |hole|  _r__|  _y__|  _b__|
                       
                       |_g__|  _b__|  _r__|  hole|
                       
                       |_r__|  hole|  _g__|  _y__|
                       
                       |hole|  _g__|  _y__|  _b__|

 'r' = red, 'b' = blue, 'g' = green and 'y' = yellow
 ------M.F.Saadi, Zanjan University, Computer Department, 11.19.2016, mfsadi@znu.ac.ir----
 * Viterbi code added on 11/25/2016, output goes to out.data, input and output files must be in 
 the current directory beside the program.Input files are: robot_train.data and robot_test.data.
