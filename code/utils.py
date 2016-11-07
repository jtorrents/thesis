#!/usr/bin/env python
#-*- coding: utf-8 -*-
# Jordi Torrents <jtorrents@milnou.net>
# Some useful functions
from __future__ import division, print_function

import math
import pickle
from itertools import count
from operator import itemgetter


##
## Read write stuff
##
# Write results in pickle format
def write_results_pkl(results, filename):
    with open(filename, 'wb') as f:
        pickle.dump(results, f)


# Load results from pickle
def load_result_pkl(filename):
    with  open(filename , 'rb') as f:
        result = pickle.load(f)
    return result


##
## ID generator
##
class UniqueIdGenerator(object):
    """A dictionary-like class that can be used to assign unique integer IDs to
    names.
    Adapted from Igraph
    """
    def __init__(self, id_generator=None, ids=None):
        """Creates a new unique ID generator. `id_generator` specifies how do we
        assign new IDs to elements that do not have an ID yet. If it is `None`,
        elements will be assigned integer identifiers starting from 0. If it is
        an integer, elements will be assigned identifiers starting from the given
        integer. If it is an iterator or generator, its `next` method will be
        called every time a new ID is needed."""
        if id_generator is None:
            id_generator = 0
        if isinstance(id_generator, int):
            self._generator = count(id_generator)
        else:
            self._generator = id_generator
        # ids is a dictionary with ids already in the database
        if not ids:
            self._ids = {}
        else:
            self._ids = ids

    def __getitem__(self, item):
        """Retrieves the ID corresponding to `item`. Generates a new ID for `item`
        if it is the first time we request an ID for it."""
        try:
            return self._ids[item]
        except KeyError:
            self._ids[item] = self._generator.next()
            return self._ids[item]

    def __len__(self):
        """Retrieves the number of added elements in this UniqueIDGenerator"""
        return len(self._ids)

    def reverse_dict(self):
        """Returns the reversed mapping, i.e., the one that maps generated IDs to their
        corresponding items"""
        return dict((v, k) for k, v in self._ids.iteritems())

    def values(self):
        """Returns the list of items added so far. Items are ordered according to
        the standard sorting order of their keys, so the values will be exactly
        in the same order they were added if the ID generator generates IDs in
        ascending order. This hold, for instance, to numeric ID generators that
        assign integers starting from a given number."""
        return sorted(self._ids.keys(), key = self._ids.__getitem__)

    def items(self):
        return sorted(self._ids.items(), key = itemgetter(1))

