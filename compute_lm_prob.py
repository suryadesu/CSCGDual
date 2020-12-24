
from lm.lm_prob import LMProb
from multiprocessing import Pool
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
import lm.model
lms = ['language_models/model.pt','language_models/modelNL.pt']
read_file_paths = ['data/java-small/train.token.code', 'data/java-small/train.token.nl']
dicts = ['data/java-small/dict_code.pkl', 'data/java-small/dict_nl.pkl']
write_file_paths = ['data/java-small/train.token.code.score', 'data/java-small/train.token.nl.score']
def get_score(line, num):
    sent = line.strip().split(' ')
    lm_score = lm_model.get_prob(sent)
    return (num, lm_score)

for i in range(2):
    print(lms[i])
    lm_model = LMProb(lms[i], dicts[i])
    fw = open(write_file_paths[i], 'w')
    f = open(read_file_paths[i])
    lines = f.readlines()
    f.close()
    with ProcessPoolExecutor(max_workers=8) as executor:
        results = executor.map(get_score, lines, list(range(len(lines))))
    scores = {}
    for result in results:
        scores[result[0]] = result[1]
    for i in range(len(lines)):
        fw.write(str(scores[i]))
        fw.write('\n')
    fw.close()


