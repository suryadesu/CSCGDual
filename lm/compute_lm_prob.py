import os
from lm_prob_cpu import LMProb
# import torch.multiprocessing as mp
from multiprocessing import Pool
import torch
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
import model
from itertools import repeat
import sys
# torch.multiprocessing.set_start_method('spawn')
lms = ['../language_models/java/model.code.pt','../language_models/java/model.nl.pt']
read_file_paths = ['../data/java/train.token.code', '../data/java/train.token.nl']
dicts = ['../data/java/dict_code.pkl', '../data/java/dict_nl.pkl']
write_file_paths = ['../data/java/train.token.code.score', '../data/java/train.token.nl.score']
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
def get_score(line, num):
    # print("enterd")
    sent = line.strip().split(' ')
    lm_score = lm_model.get_prob(sent)
    # print(lm_score)
    return (num, lm_score)
# def main():
# for i in range(2):
sel = int(sys.argv[1])
print(lms[sel])
lm_model = LMProb(lms[sel], dicts[sel])
fw = open(write_file_paths[sel], 'w')
f = open(read_file_paths[sel])
lines = f.readlines()
# k = len(lines)//2
# print(k)
# lines = lines[:k]
f.close()
scores = {}
# for i,line in enumerate(lines):
#    result = get_score(line,i)
with ProcessPoolExecutor(max_workers=10) as executor:
    results = executor.map(get_score, lines, list(range(len(lines))))
# p = Pool(4)
# results = p.starmap(get_score,zip(repeat(lm_model),lines,list(range(len(lines)))))
# results= mp.spawn(get_score,args=(lines,list(range(len(lines)))),nprocs=4,join=True)
# print(next(results))
# scores = {}
# print(results)
for result in results:
    scores[result[0]] = result[1]
    # if i %1000 == 0:
    #    print(i)
for i in range(len(lines)):
    fw.write(str(scores[i]))
    fw.write('\n')
fw.close()
#if __name__=="__main__":
#    freeze_support()
#    main()
