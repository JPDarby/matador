# coding: utf-8
""" This module contains functionality to update
and overwrite database entries with specific tasks,
e.g. symmetry and substructure analysis.
"""
from __future__ import print_function
from traceback import print_exc

from sys import exit
from utils.print_utils import print_notify, print_warning, print_failure


class Refiner:

    def __init__(self, cursor, collection=None, task=None, mode='display', **kwargs):
        possible_tasks = ['sym', 'spg',
                          'substruc', 'sub', 'remove']
        possible_modes = ['display', 'overwrite', 'set']
        if mode not in possible_modes:
            print('Mode not understood, defaulting to "display".')
            mode = 'display'
        if collection is None and mode in ['overwrite', 'set']:
            exit('Impossible to overwite or set without db collection, exiting...')
        if task is None:
            exit('No specified task, exiting...')
        elif task not in possible_tasks:
            exit('Did not understand task, please choose one of ' + ', '.join(possible_tasks))

        self.cursor = cursor
        self.diff_cursor = []
        self.collection = collection
        self.mode = mode
        self.changed_count = 0
        self.failed_count = 0
        self.args = kwargs

        if task == 'spg' or task == 'sym':
            if kwargs.get('symprec'):
                self.symmetry(symprec=kwargs.get('symprec'))
            else:
                self.symmetry()
            self.field = 'space_group'
        elif task == 'substruc' or task == 'sub':
            self.substruc()
            self.field = 'substruc'
        try:
            self.cursor.close()
        except:
            pass
        print(self.changed_count, '/', len(self.cursor), 'to be changed.')
        print(self.failed_count, '/', len(self.cursor), 'failed.')

        if self.mode in ['set', 'overwrite']:
            self.update_docs()

    def update_docs(self):
        import pymongo as pm
        requests = []
        # if in "set" mode, do not overwrite, just apply
        if self.mode == 'set':
            for ind, doc in enumerate(self.diff_cursor):
                requests.append(pm.UpdateOne({'_id': doc['_id'], self.field: {'$exists': False}}, {'$set': {self.field: doc[self.field]}}))
        # else if in overwrite mode, overwrite previous field
        elif self.mode == 'overwrite':
            for ind, doc in enumerate(self.diff_cursor):
                requests.append(pm.UpdateOne({'_id': doc['_id']}, {'$set': {self.field: doc[self.field]}}))
        if self.args.get('debug'):
            for request in requests:
                print(request)

        result = self.collection.bulk_write(requests)
        print_notify(str(result.modified_count) + ' docs modified.')

    def substruc(self):
        print('Performing substructure analysis...')
        from voronoi_interface import get_voronoi_substructure
        for ind, doc in enumerate(self.cursor):
            try:
                self.changed_count += 1
                self.cursor[ind]['substruc'] = get_voronoi_substructure(doc)
            except:
                self.failed_count += 1
                self.cursor[ind]['substruc'] = 'xxx'
                if self.args.get('debug'):
                    print_exc()
                    print_failure('Failed for' + ' '.join(doc['text_id']))
                pass
        if self.mode == 'display':
            for doc in self.cursor:
                print(doc['substruc'])

    def symmetry(self, symprec=1e-3):
        import spglib as spg
        from utils.cell_utils import doc2spg
        print('Refining symmetries...')
        if self.mode == 'display':
            print_warning('{}'.format('At symprec: ' + str(symprec)))
            print_warning("{:^36}{:^16}{:^16}".format('text_id', 'new sg', 'old sg'))
        for ind, doc in enumerate(self.cursor):
            try:
                spg_cell = doc2spg(doc)
                sg = spg.get_spacegroup(spg_cell, symprec=symprec).split(' ')[0]
                if sg != doc['space_group']:
                    self.changed_count += 1
                    self.diff_cursor.append(doc)
                    if self.mode == 'display':
                        print_notify("{:^36}{:^16}{:^16}".format(doc['text_id'][0]+' '+doc['text_id'][1], sg, doc['space_group']))
                    doc['space_group'] = sg
                else:
                    if self.mode == 'display':
                        print("{:^36}{:^16}{:^16}".format(doc['text_id'][0]+' '+doc['text_id'][1], sg, doc['space_group']))
            except:
                self.failed_count += 1
                if self.args.get('debug'):
                    print_exc()
                    print_failure('Failed for' + ' '.join(doc['text_id']))
                pass
