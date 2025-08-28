import numpy as np

import task_seqs


class Task(object):
    # primitive task class that is built upon flip and twist angles and velocities (kinematic tasks)
    # all other tasks (tricks, transitions and full combos) are sequences constructed from these primitives
    def __init__(self, name='', segms=[]):
        self.name = name
        self.abbrev_name = name
        self.params = {'phi': np.nan, 'psi': np.nan, 'dphi': np.nan, 'dpsi': np.nan}
        self.segms = segms

        #basic feet contacts
        if name=='complete':
            self.abbrev_name = 'C'
            self.segms = ['F1']
            self.add_contact(0)
        elif name=='hyper':
            self.abbrev_name = 'H'
            self.segms = ['F2']
            self.add_contact(0)
        elif name=='mega':
            self.abbrev_name = 'M'
            self.segms = ['F1']
            self.add_contact(0.5)
        elif name=='semi':
            self.abbrev_name = 'S'
            self.segms = ['F2']
            self.add_contact(0.5)
        elif name=='backside':
            self.segms = ['F1', 'F2']
            self.add_contact(0)
        elif name=='frontside':
            self.segms = ['F1', 'F2']
            self.add_contact(0.5)

        #basic hand contacts
        # be aware that n-pose is standing with palms down, and that psi rotation direction from shoulder towards hand
        # therefore we just specified which hand contacts, the orientation of the hand follows from that
        # (the in and out naming may be added later)
        elif name=='H1':
            self.segms = ['H1']
            self.add_contact()
        elif name=='H2':
            self.segms = ['H2']
            self.add_contact()

        '''
        elif name=='H1_out':
            self.segms = ['H1']
            self.add_contact(np.pi)
        elif name=='H2_out':
            self.segms = ['H2']
            self.add_contact(-np.pi)
        elif name=='H1_in':
            self.segms = ['H1']
            self.add_contact(np.pi)
        elif name=='H2_in':
            self.segms = ['H2']
            self.add_contact(-np.pi)
        '''
    def get_dict(self):
        return {'name':self.name,'abbrev_name':self.abbrev_name,'segms':self.segms,'params':self.params}

    #methods for custom primitive tasks construction
    def add_rot(self, phi, psi, dphi=0, dpsi=0):
        self.params['phi'] = phi
        self.params['psi'] = psi
        self.params['dphi'] = dphi
        self.params['dpsi'] = dpsi

    def add_contact(self, psi=np.nan):
        self.params['phi'] = 0
        self.params['psi'] = psi
        self.params['dphi'] = 0
        self.params['dpsi'] = 0

    def add_kick(self, psi, dphi):
        self.params['phi'] = 0.5
        self.params['psi'] = psi
        self.params['dphi'] = dphi
        self.params['dpsi'] = 0

    def add_flip(self, dphi):
        self.segms = ['F1', 'F2']
        self.params['dphi'] = dphi

    def add_twist(self, dpsi):
        self.segms = ['F1', 'F2']
        self.params['dpsi'] = dpsi

    def add_twisting_flip(self, dphi, dpsi):
        self.add_flip(dphi)
        self.add_twist(dpsi)

    # basic kicks
    def hook(self, dphi):
        return Task(name='hook',segms=['F1']).add_kick(0.5, dphi)

    def round(self, dphi):
        return Task(name='round', segms=['F2']).add_kick(0, dphi)

    def outside(self, dphi):
        return Task(name='outside kick', segms=['F1']).add_kick(0.25, dphi)

    def inside(self, dphi):
        return Task(name='inside kick', segms=['F2']).add_kick(0.25, dphi)

    def dleg(self, dphi):
        return Task(name='double leg', segms=['F1','F2']).add_kick(0.25, dphi)

    # Is there a kick with foot facing towards the ground, i.e. psi=-0.25?
    # It is not named, but happens (emerges) often during "inside" inversion (see loopkicks.com) tricks like aerial

class Sequence(object):
    #Methods for:
    #- assembling primitive task list from list of segms and rotations
    #- condensing primitive task list into termed sequence
    #- dissection of termed sequence into primitive task sequence
    #- primitive task sequence into list of segms and rotations

    def __init__(self):
        self.segms = []
        self.prim_names = []
        self.params = {'Dphis':[], 'Dpsis':[], 'omega_phis':[], 'omega_psis':[]}

    def from_term_list(self, term_list, name=''):
        if name=='':
            self.name = ' - '.join(term_list)
        else:
            self.name = name
        for i in range(len(term_list)):
            prim_seq = task_seqs.termed_sequences[term_list[i]]
            if i==0:
                self.segms += prim_seq['segms']
                self.params['Dphis'] += prim_seq['Dphis']
                self.params['Dpsis'] += prim_seq['Dpsis']
            else:
                if ((prim_seq['segms'][0] == self.segms[-1])
                        and (prim_seq['Dpsis'][0] % 1==self.params['Dphis'][-1] % 1)
                        and (prim_seq['Dpsis'][0] % 1==self.params['Dpsis'][-1] % 1)):
                    self.segms += prim_seq['segms'][1:]
                    self.params['Dphis'] += prim_seq['Dphis'][1:]
                    self.params['Dpsis'] += prim_seq['Dpsis'][1:]
                else:
                    print('Linkage between',term_list[i-1],'and',term_list[i],'not possible.')
                    continue


#defining extremities in both ways (should be detected automatically from negative/positive twist component of L_G)
def ex_dict(tricking_side):
    if tricking_side=='orthodox':
        return {'F1':'LeftFoot','F2':'RightFoot','H1':'LeftHand','H2':'RightHand','H':'Head'}
    if tricking_side=='unorthodox':
        return {'F1':'RightFoot','F2':'LeftFoot','H1':'RightHand','H2':'LeftHand','H':'Head'}