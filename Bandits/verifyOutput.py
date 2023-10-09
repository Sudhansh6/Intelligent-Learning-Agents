import os,random,subprocess
random.seed(2)
fname= "outputData.txt"
algo_ls = ['epsilon-greedy-t1','ucb-t1','kl-ucb-t1','thompson-sampling-t1','ucb-t2','alg-t3','alg-t4']
number_lines = 9150

errorFlag = False



print('\n\n-------- verifying', fname ,'data ---------------')
with open(fname, "r") as f:
    line_ls = [line.replace("\n","").replace(", ",",") for line in f if line!="\n"]
    len_line_ls = len(line_ls)

    #Check for number of lines
    if not (len_line_ls==number_lines):
        print("\n","*"*10,"Mistake:number of lines in the output data file should be",number_lines,"but have ",len_line_ls,"*"*10,"\n")
        errorFlag = True


    lists = []
    set_main=set()
    set_algos = set()
    for i in range(9):
        lists.append([])

    for line in line_ls:
        line = line.split(",")
        if not len(line)==9:
            print("\n","*"*10,"Mistake: Wrong line printed",line,len(line),"*"*10,"\n")
            continue
        lists[0].append(line[0])  #instance
        lists[1].append(line[1])  #algo
        lists[2].append(int(line[2])) #randomSeed
        lists[3].append(float(line[3])) #epsilon
        lists[4].append(float(line[4])) #scale
        lists[5].append(float(line[5])) #threshold
        lists[6].append(int(line[6])) # horizon
        lists[7].append(float(line[7])) # REG
        lists[8].append(float(line[8])) # HIFHS


        set_main.add(line[0]+"--"+line[1]+"--"+line[2]+"--"+line[3]+"--"+line[4]+"--"+line[5]+"--"+line[6])
        set_algos.add(line[1])
    if not len(set_algos)==7:
        print("You have implemented only:", set_algos)
        errorFlag = True
    if not len(set_main)==number_lines:
        print("\n","*"*10,"Mistake: You didn't print all the combinations. Need ",number_lines,"but printed ",len(set_main),"*"*10,"\n")
        errorFlag = True

    for i in range(10):
        line_str = line_ls[random.randint(0,len_line_ls)]
        line = line_str.replace("\n","").split(",")

        orig_REG = line[-2].strip()
        orig_HIGHS = line[-1].strip()

        cmd = "python","bandit.py","--instance",line[0].strip(),"--algorithm",line[1].strip(),"--randomSeed",line[2].strip(),"--epsilon",line[3].strip(),"--scale",line[4].strip(),"--threshold",line[5].strip(),"--horizon",line[6].strip()
        print("running",cmd,end="\t")
        reproduced_str = subprocess.check_output(cmd,universal_newlines=True)
        reproduced = reproduced_str.replace("\n","").split(",")
        rep_REG = reproduced[-2].strip()
        rep_HIGH = reproduced[-1].strip()

        if not rep_REG==orig_REG:
            print("\n","*"*10,"Mistake: Unable to reproduce result for REG ",line_str," orignal="+orig_REG+" reproduced="+rep_REG,"\t","*"*10,"\n")
            #print(line_str)
            errorFlag = True
        else:
            print("REG checked OK",end="\t")
        if not orig_HIGHS==rep_HIGH:
            print("\n","*"*10,"Mistake: Unable to reproduce result for HIGHS ",line_str," orignal="+orig_HIGHS+" reproduced="+rep_HIGH,"\t","*"*10,"\n")
            #print(line_str)
            errorFlag = True
        else:
            print("HIGHS checked OK")


if errorFlag:
    print("\n","*"*10,"Some issue with your submission data","*"*10,"\n")
else:
    print("Everything is Okay")
