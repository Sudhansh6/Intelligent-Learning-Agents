import os
import numpy as np
import matplotlib.pyplot as plt

np.random.seed(0)

statesFile1 = "data/attt/states/states_file_p1.txt"
statesFile2 = "data/attt/states/states_file_p2.txt"

with open(statesFile1) as f:
    fileContent = f.readlines()
states1 = [line[:-1] for line in fileContent]
with open(statesFile2) as f:
    fileContent = f.readlines()
states2 = [line[:-1] for line in fileContent]


# write initial policy to file
os.system('cp data/attt/policies/p2_policy1.txt task3/p2_policy_0')
os.system('cp data/attt/policies/p1_policy1.txt task3/p1_policy_0')


p1wr = []
p2wr = []

loops = 10
iter_no = 1
x = []

def train_p1(i):
	# Get P1
	print("Running encoder for p1")
	os.system('python encoder.py --policy task3/p2_policy_' + str(i) + \
	 ' --states data/attt/states/states_file_p1.txt > task3/p1_mdp')
	print("Running planner for p1")
	os.system('python planner.py --mdp \
		task3/p1_mdp > task3/p1_vp')
	print("Running decoder for p1")
	os.system('python decoder.py --value-policy \
		task3/p1_vp --states data/attt/states/states_file_p1.txt \
		--player-id 1 > task3/p1_policy_' + str(i + 1))

def train_p2(i):
	print("Running encoder for p2")
	os.system('python encoder.py --policy task3/p1_policy_' + str(i + 1) + \
	 ' --states data/attt/states/states_file_p2.txt > task3/p2_mdp')
	print("Running planner for p2")
	os.system('python planner.py --mdp \
		task3/p2_mdp > task3/p2_vp')
	print("Running decoder for p2")	
	os.system('python decoder.py --value-policy \
		task3/p2_vp --states data/attt/states/states_file_p2.txt \
		--player-id 2 > task3/p2_policy_' + str(i + 1))

p1diff = [0]*loops
p2diff = [0]*loops

# Now that initialisation is done, iterate
for i in range(loops):
	print(i)

	x.append(i)
	train_p1(i)
	train_p2(i)

	# Get number of different lines in policy files to check convergence
	p1diff[i] = int(os.popen('diff -y --suppress-common-lines task3/p1_policy_'\
		+ str(i + 1) + ' task3/p1_policy_' + str(i) + '| wc -l').read())
	p2diff[i] = int(os.popen('diff -y --suppress-common-lines task3/p2_policy_'\
		+ str(i + 1) + ' task3/p2_policy_' + str(i) + '| wc -l').read())

plt.plot(x, p1diff)
plt.plot(x, p2diff)
plt.legend(['Player1', 'Player2'])
plt.title('Difference in policy files of Player1 and Player2')
plt.xlabel('Training iterations')
plt.ylabel('Number of differences')
plt.show()
plt.savefig("plot.png")
