import sys

if len(sys.argv) == 3:
    print ('You typed in 2:{} 1:{}'.format(sys.argv[2], sys.argv[1]))
else:
    print ('invalid number of args. argv=[{}]'.format(str(sys.argv)))