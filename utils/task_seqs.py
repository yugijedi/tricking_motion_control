# there is no clear separation between contact sequences and flight sequences in tricking terminology
# here "tricks" are flight sequence and "transitions" and "setups" are contact sequences
# also

#from combo_graph import segms

#How to count rotation if the task segment group changes?
# Because the task segm groups are only extremities, we
# (1) place the current task segm group at the orientation (phi, psi) of the last task segm group
# (2) rotate the current task segm group by Dphi to the next task orientation, while respecting Dphi>=0
# (3) rotate the current task segm group by Dpsi to the next task orientation, while respecting Dpsi>=0
# If dtheta=0 and only half-integer differences between tasks are considered, (2) and (3) commute, so that rotation order doesn't matter.
# However, (1) does not commute with both (2) and (3). This means it has to be the first operation we do.
# The "segm replacement" (1) is a simplification that makes it possible to specify only one sequence of rotation differences.
# Its disadvantage is that the actual cumulative rotation that all segms completed is of course different from the cumulated sequence of rotation differences.
# We don't need to know this cumulative rotation for naming sequences though. We would also have to track the rotation of all segms to calculate it.

termed_sequences = {
    #'term':{'segms':task segment groups as list of lists, 'Dphis':[initial phi,Dphi1,...,DphiN], 'Dpsis':[initial psi,Dpsi1,...,DpsiN], 'spec':'...'

    #NO-HANDED ONE-CONTACT TRANSITIONS
    'Backswing':{'segms':[['F1']], 'Dphis':[0], 'Dpsis':[0]},
    'Wrap/Hyperswing':{'segms':[['F2']], 'Dphis':[0], 'Dpsis':[0]},
    'Unwrap/Megaswing':{'segms':[['F1']], 'Dphis':[0], 'Dpsis':[0.5]},
    'Frontswing':{'segms':[['F2']], 'Dphis':[0], 'Dpsis':[0.5]},

    #NO-HANDED TWO-CONTACT TRANSITIONS
    'Carrythrough (C-M)':{'segms':[['F1'],['F1']], 'Dphis':[0,0], 'Dpsis':[0,0.5]},
    'Vanish':{'segms':[['F1'],['F2']], 'Dphis':[0,0], 'Dpsis':[0,0]},
    'Skip':{'segms':[['F1'],['F2']], 'Dphis':[0,0], 'Dpsis':[0,0.5]},
    'Boneless':{'segms':[['F2'],['F1']], 'Dphis':[0,0], 'Dpsis':[0,0]},
    'Turnstep':{'segms':[['F2'],['F1']], 'Dphis':[0,0], 'Dpsis':[0,0.5]},
    'Carrythrough (H-S)':{'segms':[['F2'],['F2']], 'Dphis':[0,0], 'Dpsis':[0,0.5]},
    'Front Boneless':{'segms':[['F1'],['F2']], 'Dphis':[0,0], 'Dpsis':[0.5,0]},
    'Front Vanish':{'segms':[['F2'],['F1']], 'Dphis':[0,0], 'Dpsis':[0.5,0]},

    #NO-HANDED THREE-CONTACT TRANSITIONS
    'Reversal':{'segms':[['F1'],['F2'],['F1']],'Dphis':[0,0,0], 'Dpsis':[0,0.25,0.5]},
    'Redirect':{'segms':[['F2'],['F1'],['F2']],'Dphis':[0,0,0], 'Dpsis':[0,0.25,0.5]},

    #NO-HANDED DOUBLE-CONTACT TRANSITIONS
    'Pop':{},
    'Reverse Pop':{},
    'Punch':{},
    

    #SINGLE FLIPS WITHOUT KICKS
    'Gainer Switch':{'segms':[['F1'],['F1']], 'Dphis':[0,1], 'Dpsis':[0,0]},
    'Gainer Mega/Scissor/Heli':{'segms':[['F1'],['F1']], 'Dphis':[0,1], 'Dpsis':[0,0.5]},
    'Cork':{'segms':[['F1'],['F1']],'Dphis':[0,1], 'Dpsis':[0,1]},
    'Cork Mega/Scissor/Heli':{'segms':[['F1'],['F1']],'Dphis':[0,1], 'Dpsis':[0,1.5]},
    'Double Cork':{'segms':[['F1'],['F1']],'Dphis':[0,1], 'Dpsis':[0,2]},
    'Triple Cork':{'segms':[['F1'],['F1']],'Dphis':[0,1], 'Dpsis':[0,3]},
    'Quad Cork':{'segms':[['F1'],['F1']],'Dphis':[0,1], 'Dpsis':[0,4]},

    #SINGLE FLIPS WITHOUT KICKS
    'Gainer':{'segms':[['F1'],['F2']],'Dphis':[0,1], 'Dpsis':[0,0]},
    'Gainer Semi':{'segms':[['F1'],['F2']],'Dphis':[0,1], 'Dpsis':[0,0.5]},
    'Cork Hyper':{'segms':[['F1'],['F2']],'Dphis':[0,1], 'Dpsis':[0,1]},
    'Cork Semi':{'segms':[['F1'],['F2']],'Dphis':[0,1], 'Dpsis':[0,1.5]},

    'GM Scoot':{'segms':[['F2'],['F1']],'Dphis':[0,1], 'Dpsis':[0,0]},
    'GM Swipe Mega/Scissor/Heli':{'segms':[['F2'],['F1']],'Dphis':[0,1], 'Dpsis':[0,0.5]},
    'Wrap/GM Full':{'segms':[['F2'],['F1']],'Dphis':[0,1], 'Dpsis':[0,1]},
    'Wrap/GM Full Mega':{'segms':[['F2'],['F1']],'Dphis':[0,1], 'Dpsis':[0,1.5]},

    'GM Swipe':{'segms':[['F2'],['F2']],'Dphis':[0,1], 'Dpsis':[0,0]},
    'GM Swipe Semi':{'segms':[['F2'],['F2']],'Dphis':[0,1], 'Dpsis':[0,0.5]},
    'Wrap/GM Full Hyper':{'segms':[['F2'],['F2']],'Dphis':[0,1], 'Dpsis':[0,1]},
    'Wrap/GM Full Semi':{'segms':[['F2'],['F2']],'Dphis':[0,1], 'Dpsis':[0,1.5]},

    'Aerial Mega/Scissor/Heli':{'segms':[['F1'],['F1']],'Dphis':[0,1], 'Dpsis':[0.5,0]},
    'Btwist':{'segms':[['F1'],['F1']],'Dphis':[0,1], 'Dpsis':[0.5,0.5]},
    'Btwist Mega/Scissor/Heli':{'segms':[['F1'],['F1']],'Dphis':[0,1], 'Dpsis':[0.5,1]},
    'Double Btwist':{'segms':[['F1'],['F1']],'Dphis':[0,1], 'Dpsis':[0.5,1.5]},

    'Webster (M-S)':{'segms':[['F1'],['F2']],'Dphis':[0,1], 'Dpsis':[0.5,0]},
    'Aerial':{'segms':[['F1'],['F2']],'Dphis':[0,1], 'Dpsis':[0.5,0.5]},
    'Aerial Semi':{'segms':[['F1'],['F2']],'Dphis':[0,1], 'Dpsis':[0.5,1]},
    'Btwist Hyper':{'segms':[['F1'],['F2']],'Dphis':[0,1], 'Dpsis':[0.5,1.5]},

    'Webster (S-M)':{'segms':[['F2'],['F1']],'Dphis':[0,1], 'Dpsis':[0.5,0]},
    'Raiz':{'segms':[['F2'],['F1']],'Dphis':[0,1], 'Dpsis':[0.5,0.5]},
    'Raiz Mega/Scissor/Heli':{'segms':[['F2'],['F1']],'Dphis':[0,1], 'Dpsis':[0.5,1]},
    'Front Cork Complete':{'segms':[['F2'],['F1']],'Dphis':[0,1], 'Dpsis':[0.5,1.5]},

    'Webster (S-S)':{'segms':[['F2'],['F2']],'Dphis':[0,1], 'Dpsis':[0.5,0]},
    'Sideswipe':{'segms':[['F2'],['F2']],'Dphis':[0,1], 'Dpsis':[0.5,0.5]},
    'Sideswipe/Front Cork Semi':{'segms':[['F2'],['F2']],'Dphis':[0,1], 'Dpsis':[0.5,1]},
    'Front Cork Hyper':{'segms':[['F2'],['F2']],'Dphis':[0,1], 'Dpsis':[0.5,1.5]},

    #BASIC KICKS
    'Round':{'segms':[['F2']],'Dphis':[0.5], 'Dpsis':[0]},
    'Hook':{'segms':[['F1']],'Dphis':[0.5], 'Dpsis':[0.5]},
    'Inside':{'segms':[['F2']],'Dphis':[0.5], 'Dpsis':[-0.25]},
    'Outside':{'segms':[['F1']],'Dphis':[0.5], 'Dpsis':[-0.25]}, #=Shuriken
    'DLeg':{'segms':[['F1','F2']],'Dphis':[0.5],'Dpsis':[-0.25]},

    #Compound Kicks are defined by takeoff stance. It is possible to land them in different stances, however the name does not always change.
    #--> We should always specify the takeoff, but not always the landing

    #CHEAT KICKS (takes off in Hyper and lands in Mega or Hyper)
    'Cheat 360':{'segms':[['F2'],['F2']],'Dphis':[0,0.5],'Dpsis':[0,0]},
    #540=360 Hyper, F2 only flips from H into H.
    'Cheat 720':{'segms':[['F2'],['F1']],'Dphis':[0,0.5],'Dpsis':[0,0.5]},
    'Cheat 900':{'segms':[['F2'],['F2']],'Dphis':[0,0.5],'Dpsis':[0,1]},
    'Cheat 1080':{'segms':[['F2'],['F1']],'Dphis':[0,0.5],'Dpsis':[0,1.5]},
    'Cheat 1260':{'segms':[['F2'],['F2']],'Dphis':[0,0.5],'Dpsis':[0,2]},
    'Cheat 1440':{'segms':[['F2'],['F1']],'Dphis':[0,0.5],'Dpsis':[0,2.5]},
    'Cheat 1620':{'segms':[['F2'],['F2']],'Dphis':[0,0.5],'Dpsis':[0,3]},

    'Parafuso':{'segms':[['F2'],['F1','F2']],'Dphis':[0,0.5],'Dpsis':[0,-0.25]}, #DLeg taking off in Hyper

    'Feilong':{'segms':[['F1'],['F2']],'Dphis':[0.5,0], 'Dpsis':[-0.25,0]}, #=Outside-Inside

    'Swipeknife':{'segms':[['F2'],['F1']],'Dphis':[0.5,0],'Dpsis':[0,0.5]}, #=Round-Hook (Jackknife)
    '540 Double':{'segms':[['F2'],['F2']],'Dphis':[0.5,1],'Dpsis':[0,-0.25]}, #=Round-Inside
    '720 Double':{'segms':[['F1'],['F1']],'Dphis':[0.5,1],'Dpsis':[-0.25,0]}, #=Outside-Outside
    'Shurikencutter':{'segms':[['F1'],['F1']],'Dphis':[0.5,1],'Dpsis':[-0.25,-0.25]}, #=Outside-Hook
    'Hurricane Kick':{'segms':[['F1'],['F1'],['F1']],'Dphis':[0.5,1,1],'Dpsis':[0,-0.25,0]}, #=Outside-Outside-Outside
    'Shurikane':{'segms':[['F1'],['F1'],['F1']],'Dphis':[0.5,1,1],'Dpsis':[-0.25,0,-0.25]}, #=Outside-Outside-Hook

}

task_seq_old = {('BSide','F1','F1',1,0):'Gainer Switch'}

#task_seq_by_rot_inv = {v: k for k, v in termed_sequences.items()}

stance_conv = {('BSide','F1','F1','full'):('C','C'),
                ('BSide','F1','F1','half'):('C','M'),
                ('BSide','F1','F2','full'):('C','H'),
                ('BSide','F1','F2','half'):('C','S'),
                ('BSide','F2','F1','full'):('H','C'),
                ('BSide','F2','F1','half'):('H','M'),
                ('BSide','F2','F2','full'):('H','H'),
                ('BSide','F2','F2','half'):('H','S'),
                ('FSide','F1','F1','full'):('M','M'),
                ('FSide','F1','F1','half'):('M','C'),
                ('FSide','F1','F2','full'):('M','S'),
                ('FSide','F1','F2','half'):('M','H'),
                ('FSide','F2','F1','full'):('S','M'),
                ('FSide','F2','F1','half'):('S','C'),
                ('FSide','F2','F2','full'):('S','S'),
                ('FSide','F2','F2','half'):('S','H')}

def task_seq_by_stance(inverse=False):
    dict = {}
    for key, value in task_seq_by_rot.items():
        if key[4]%1 == 0: #full twist rotation
            rot = 'full'
        elif key[4]%1 == 0.5:
            rot = 'half'
        else:
            print('Provide integer or half-integer rotation numbers.')
        stance_trans = stance_conv[key[0],key[1],key[2],rot]
        dict.update({(stance_trans[0],stance_trans[1],key[3],key[4]):value})
    if inverse==True:
        return {v: k for k, v in dict.items()}
    else:
        return dict

old = {
    'backswing':['complete'],
    'hyper-backswing':['hyper'],
    'mega-frontswing':['mega'],
    'frontswing':['semi'],
    #double foot contacts:
    # these terms do actually not depend on the start and end stance in the usual way,
    # but in my opinion they should be used in this way.
    'vanish':['complete','hyper'],
    'skip':['complete','semi'],
    'boneless':['hyper','complete'],
    'carrythrough':['hyper','semi'],
    #triple foot contacts
    # the middle contact takes almost no force and is in some undefined stance in between
    'reversal':['complete','','mega'],
    'redirect':['hyper','','semi'],
    #incl. hand contacts (setups)
    'scoot':['hyper','H1','complete'],
    'masterscoot':['hyper','H1','H2','complete'],
    'palmkick':['mega','H2','hyper'],
    'cartwheel':['mega','H1','H2','hyper'],
    'left-handed cartwheel':['mega','H1','hyper'],
    'gumbi':['semi','H2','H1','complete'],
    'touchdown raiz':['semi','H1','complete'],
    'macaco':['backside','H1','complete']}

flight_seqs = {

}