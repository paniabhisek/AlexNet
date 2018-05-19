#!/usr/bin/python3
# -*- coding: utf-8 -*-

import logging

def get_logger():
    logger = logging.getLogger('AlexNet')
    logger.setLevel(logging.DEBUG)

    fh = logging.FileHandler('logs.log')
    fh.setLevel(logging.DEBUG)

    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)

    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    fh.setFormatter(formatter)
    ch.setFormatter(formatter)

    logger.addHandler(fh)
    logger.addHandler(ch)

    return logger
