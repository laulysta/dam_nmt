import sys, os
from subprocess import check_output, call
import argparse
import fcntl

# Parse the arguments
parser = argparse.ArgumentParser()
parser.add_argument('-m', '--model_path', required=True, help='The path of the model')
parser.add_argument('-vs', '--vocab_src_path', required=True, help='The path of the source dictionary')
parser.add_argument('-vt', '--vocab_trg_path', required=True, help='The path of the target dictionary')
parser.add_argument('-ts', '--txt_src_path', required=True, help='The path of the source text reference')
parser.add_argument('-tt', '--txt_trg_path', required=True, help='The path of the target text reference')
#parser.add_argument('-mv', '--model_version', required=True, help='the normal nmt version or the nmt_bow version (ex: nmt, nmt_bow)')
args = parser.parse_args()

def createDir(dirPath):
    if not os.path.exists(dirPath):
        try:
            os.makedirs(dirPath)
        except OSError as e:
            print e
            print 'Exeption was catch, will continue script \n'


def compute_translation_and_bleu(model_path, vocab_src_path, vocab_trg_path, txt_src_path, txt_trg_path, txt_trans_result_path):

    translateFilePath = '../nmt/translate.py'

    #check_output('python ' + translateFilePath + ' -r 0.5 -n ' + model_path + ' ' + vocab_src_path + ' ' + vocab_trg_path + ' ' + txt_src_path + ' ' + txt_trans_result_path, shell=True)
    call('python -u ' + translateFilePath + ' -p 7 ' + ' -n ' + model_path + ' ' + vocab_src_path + ' ' + vocab_trg_path + ' ' + txt_src_path + ' ' + txt_trans_result_path, shell=True)

    bleu_score = check_output('perl multi-bleu.perl ' + txt_trg_path + ' < ' +  txt_trans_result_path , shell=True)

    return bleu_score


model_name = args.model_path.split('/')[-2]
model_name_with_epoch = args.model_path.split('/')[-1]

savePathDir = 'translation_results_' + model_name + '/'
createDir(savePathDir)

savePathTranslationsDir = savePathDir + 'translations/'
createDir(savePathTranslationsDir)

result_file_name = (args.txt_src_path.split('/')[-1]).split('.')[0]
trg_lang = (args.txt_trg_path.split('/')[-1]).split('.')[-1]
txt_trans_result_path = savePathTranslationsDir + '/' + 'translation_' + model_name_with_epoch + '_' + result_file_name + '.' + trg_lang
file_bleu_result_path = savePathDir + 'results_bleu_' + result_file_name


bleu_score = compute_translation_and_bleu(args.model_path, args.vocab_src_path, args.vocab_trg_path, args.txt_src_path, args.txt_trg_path, txt_trans_result_path)


f = open(file_bleu_result_path, "a")
fcntl.flock(f.fileno(), fcntl.LOCK_EX)
f.write(model_name_with_epoch + ' :   ' + bleu_score + '\n')
f.close()
