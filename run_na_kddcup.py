import argparse
import src.param as param
import os
from os.path import join
import numpy as np
import pandas as pd
import logging
from src.data_loader import load_support_kgs, load_target_kg, load_all_to_all_seed_align_links
from src.validate import MultiModelTester, TestMode, test_alignment_hits10
from src.model import save_model_structure


def set_logger(param, model_dir):
    '''
    Write logs to checkpoint and console
    '''

    log_file = os.path.join(model_dir, 'train.log')

    logging.basicConfig(
        format='%(asctime)s %(levelname)-8s %(message)s',
        level=logging.INFO,
        datefmt='%Y-%m-%d %H:%M:%S',
        filename=log_file,
        filemode='w'
    )
    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s %(levelname)-8s %(message)s')
    console.setFormatter(formatter)
    logging.getLogger('').addHandler(console)


def parse_args(args=None):
    parser = argparse.ArgumentParser(
        description='Training and Testing Knowledge Graph Embedding Models',
        usage='run.py [<args>] [-h | --help]'
    )
    parser.add_argument('-l', '--target_language', default="mag", type=str, choices=['ja', 'el', 'es', 'fr', 'en'],
                        help="target kg")
    parser.add_argument('-m', '--knowledge_model', default='transe', type=str, choices=['transe', 'rotate'])
    parser.add_argument('--use_default', action="store_true",
                        help="Use default setting. This will override every setting except for targe_langauge and knowledge_model")
    parser.add_argument('-lr', '--learning_rate', default=1e-2, type=float, help="learning rate for knowledge model")
    parser.add_argument('--align_lr', default=1e-3, type=float, help="learning rate for knowledge model")
    parser.add_argument('-b', '--batch_size', default=1024, type=int, help="batch size of queries")
    parser.add_argument('-d', '--dim', default=400, type=int)
    parser.add_argument('--transe_margin', default=0.3, type=float)
    parser.add_argument('--rotate_gamma', default=24, type=float)
    parser.add_argument('--reg_scale', default=1e-5, type=float, help="scale for regularization")
    parser.add_argument('--base_align_step', default=5, type=float,
                        help="how many align model epoch to train before switching to knowledge model")
    parser.add_argument('--knowledge_step_ratio', default=2, type=float,
                        help="how many knowledge model epochs for each align epoch")
    parser.add_argument('--round', default=5, type=float,
                        help="how many rounds to train")
    parser.add_argument('-k', default=10, type=int, help="how many nominations to consider")

    return parser.parse_args(args)


def set_params(args):
    param.lang = args.target_language
    param.knowledge = args.knowledge_model

    if args.use_default:
        if param.knowledge == 'transe':
            param.epoch10 = 10
            param.epoch11 = 10
            param.epoch2 = 5
            param.lr = 1e-3
            param.dim = 32
            param.round = 2
        elif param.knowledge == 'rotate':
            param.epoch10 = 100
            param.epoch11 = 100
            param.epoch2 = 5
            param.lr = 1e-2
            param.dim = 32
            param.round = 3
    else:
        param.dim = args.dim
        param.lr = args.learning_rate
        param.batch_size = args.batch_size
        param.epoch2 = args.base_align_step
        param.epoch10 = param.epoch11 = args.base_align_step * args.knowledge_step_ratio
        param.gamma = args.rotate_gamma
        param.margin = args.transe_margin
        param.reg_scale = args.reg_scale
        param.round = args.round



def main(args):
    set_params(args)

    target_lang = param.lang
    # src_langs = ['fr', 'ja', 'es', 'el', 'en']
    src_langs = ['aminer', 'mag']
    src_langs.remove(target_lang)

    target_lang = param.lang

    # load data
    data_dir = '/home/zfj/research-data/na-checking/kens-na-data-kddcup/kg'  # where you put kg data
    seed_dir = '/home/zfj/research-data/na-checking/kens-na-data-kddcup/seed_alignlinks'  # where you put seed align links data
    model_dir = join('./trained_model', f'kens-{param.knowledge}-{param.dim}', target_lang)  # output
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)

    # logging
    set_logger(param, model_dir)  # set logger
    logging.info('Knowledge model: %s' % (param.knowledge))
    logging.info('target language: %s' % (param.lang))

    # hyper-parameters
    logging.info(f'dim: {param.dim}')
    logging.info(f'lr: {param.lr}')

    # target (sparse) kg
    kg0 = load_target_kg(data_dir, target_lang, testfile_suffix='-val.tsv')  # target kg. KnowledgeGraph object
    # supporter kgs
    supporter_kgs = load_support_kgs(data_dir, seed_dir, target_lang, src_langs)  # list[KnowledgeGraph]
    # supporter KG use all links to train

    # seed alignment links
    seed_alignlinks = load_all_to_all_seed_align_links(seed_dir)  # {(lang1, lang2): 2-col np.array}

    all_kgs = [kg0] + supporter_kgs

    # build alignment model (all-to-all)
    for kg in all_kgs:
        kg.build_alignment_models(all_kgs)  # kg.align_models_of_all {lang: align_model}

    # create validator
    validator = MultiModelTester(kg0, supporter_kgs)

    print('model initialization done')

    for i in range(param.round):
        # train alignment model
        logging.info("training alignment model...")
        for kg in all_kgs:
            # align it with everything else
            for other_lang, align_model in kg.align_models_of_all.items():
                if (other_lang, kg.lang) in seed_alignlinks:  # seed_alignlinks {(lang1, lang2): 2-col np.array}
                    # use if to avoid retrain the same pair of languages
                    align_links = seed_alignlinks[(other_lang, kg.lang)]
                    align_model.fit([align_links[:, 0], align_links[:, 1]], np.zeros(align_links.shape[0]),
                                    epochs=param.epoch2, batch_size=param.batch_size, shuffle=True, )

        print("\n\n")
        logging.info("**********************************")

        # train knowledge model
        logging.info("training target kg model")
        kg0.model.fit([kg0.h_train, kg0.r_train, kg0.t_train], kg0.y_train,
                      epochs=param.epoch10, batch_size=param.batch_size, shuffle=True, )
        logging.info("training support kg model")
        for kg1 in supporter_kgs:
            kg1.model.fit([kg1.h_train, kg1.r_train, kg1.t_train], kg1.y_train,
                          epochs=param.epoch11, batch_size=param.batch_size, shuffle=True, )

        model_weights = []
        if i % param.val_freq == 0:  # validation
            logging.info(f'=== round {i}')
            logging.info(f'[{kg0.lang}]')
            hits10_kg0 = validator.test(TestMode.KG0)

    kg0.save_model(model_dir)
    for kg1 in supporter_kgs:
        kg1.save_model(model_dir)

    save_model_structure(kg0.kNN_finder, os.path.join(model_dir, 'kNN_finder.json'))

    for kg1 in supporter_kgs:
        kg1.populate_alignlinks(kg0, seed_alignlinks)

    choices = ['-val.tsv', '-test.tsv']  # '-val.tsv' if predict on validation data, '-test.tsv' if on test data
    validator = MultiModelTester(kg0, supporter_kgs)

    kg0.filtered_reordered_embedding_matrix = None
    for kg1 in supporter_kgs:
        kg1.filtered_reordered_embedding_matrix = None


    for suffix in choices:
        testfile = join(data_dir, target_lang + suffix)
        output = join(model_dir, 'results-dist' + suffix)
        testcases = pd.read_csv(testfile, sep='\t', header=None).values
        param.n_test = testcases.shape[0]  # test on all training triples
        print('Loaded test cases from: %s' % testfile)

        # results_df = validator.test_and_record_results(testcases)
        dists = validator.calc_test_cases_sim(testcases)

        with open(output, "w") as wf:
            for d in dists:
                wf.write(str(d) + "\n")


if __name__ == "__main__":
    main(parse_args())
    