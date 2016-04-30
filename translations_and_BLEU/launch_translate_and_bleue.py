import sys, os
import datetime
from subprocess import check_output
import argparse
from os.path import join as pjoin

# Parse the arguments
parser = argparse.ArgumentParser()
parser.add_argument('-m', '--model_path', required=True, help='The path of the model')
parser.add_argument('-dv', '--data_version', default="small", help='the data version (ex: small, wmt)')
parser.add_argument('-o', '--out', help='Name of the file that will contain the translation commands to launch. Default: "translation_commands_for_{model_path}.txt"')
args = parser.parse_args()

def createDir(dirPath):
    if not os.path.exists(dirPath):
        try:
            os.makedirs(dirPath)
        except OSError as e:
            print e
            print 'Exeption was catch, will continue script \n'


listPathFiles = []
for (dirpath, dirname, filenames) in os.walk(args.model_path):
    fileChoosed = ''
    for f in filenames:
        if f[-4:] == '.npz':
            listPathFiles += [pjoin(dirpath, f)]




#print "smart_dispatch.py -qsw2 -t24:00:00 -c2 launch python -u translate_and_bleu.py -m [" + " ".join(listPathFiles) + "]  -vs ../data/vocab_and_data_europarl/vocabEuroparl.en.pkl -vt ../data/vocab_and_data_europarl/vocabEuroparl.fr.pkl -ts raw_wmt14_valid_test_enfr/ntst1213.en -tt raw_wmt14_valid_test_enfr/ntst1213.fr"
  #-ts ../data/raw_wmt14_valid_test_enfr/ntst1213.en -tt ../data/raw_wmt14_valid_test_enfr/ntst1213.fr

def check_model_name_in_results_file(model_name, results_file):
    if not os.path.isfile(results_file):
        return False

    with open(results_file) as f:
        for line in f:
            if len(line.strip()) == 0:
                continue

            if line.split(" ")[0] == model_name:
                return True

    return False


if args.data_version == 'small':
    vocabSource = "../data/vocab_and_data_small_europarl_v7_enfr/vocab.en.pkl"
    vocabTarget = "../data/vocab_and_data_small_europarl_v7_enfr/vocab.fr.pkl"
else:
    sys.exit('Wrong data version')


modelTrainingName = os.path.basename(os.path.dirname(args.model_path))

commands = []
for nb, modelPath in enumerate(listPathFiles):
    modelName = os.path.splitext(os.path.basename(modelPath))[0]

    save_path_valid = pjoin("translation_results_" + modelTrainingName, "results_bleu_ntst1213")
    save_path_test = pjoin("translation_results_" + modelTrainingName, "results_bleu_ntst14")

    # Check if validation results already exist for this model.
    if not check_model_name_in_results_file(modelName, save_path_valid):
        commandValid = "THEANO_FLAGS='device=cpu' python -u translate_and_bleu.py -m " + modelPath + " -vs " + vocabSource + " -vt " + vocabTarget + " -ts ../data/raw_wmt14_valid_test_enfr/ntst1213.en -tt ../data/raw_wmt14_valid_test_enfr/ntst1213.fr"
        commands.append(commandValid)

    # Check if test results already exist for this model.
    if not check_model_name_in_results_file(modelName, save_path_test):
        commandTest = "THEANO_FLAGS='device=cpu' python -u translate_and_bleu.py  -m " + modelPath + " -vs " + vocabSource + " -vt " + vocabTarget + " -ts ../data/raw_wmt14_valid_test_enfr/ntst14.en -tt ../data/raw_wmt14_valid_test_enfr/ntst14.fr"
        commands.append(commandTest)

commands_filename = "translation_commands_for_" + modelTrainingName + ".txt"
if args.out is not None:
    commands_filename = args.out

with open(commands_filename, 'w') as f:
    f.write("\n".join(commands) + "\n")
