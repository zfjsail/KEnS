from os.path import join
import json

import logging

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format='%(asctime)s %(message)s')  # include timestamp


def load_json(rfdir, rfname):
    logger.info('loading %s ...', rfname)
    with open(join(rfdir, rfname), 'r', encoding='utf-8') as rf:
        data = json.load(rf)
        logger.info('%s loaded', rfname)
        return data
