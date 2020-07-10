#!/usr/bin/env python3

import sys, os

import sys, os


if len(sys.argv) == 3:
    print ('You typed in 2:{} 1:{}'.format(sys.argv[2], sys.argv[1]))
else:
    print ('invalid number of args. argv=[{}]'.format(str(sys.argv)))

print('Bin: '+str(os.path.dirname(sys.executable)))