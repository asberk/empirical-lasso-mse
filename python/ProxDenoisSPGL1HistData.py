import pdmse
import sys
import jsonWrite


def main(logNmax, filename, logfile=sys.stdout):
    pdmse_dict = {'N': pdmse.pdmse_batch(logNmax,
                                         verbose=logfile),
                  'sqNormZ': pdmse.pdmse_batch(logNmax,
                                               theta='sqNormZ',
                                               verbose=logfile)}
    fp = open(logfile, 'a', encoding='utf-8')
    print('\nRuns complete. Saving dict to json...', file=fp, end='')
    jsonWrite.dict(filename, pdmse_dict)
    print('complete!', file=fp)
    return

if __name__ == "__main__":
    from datetime import datetime as dt
    now = dt.now().ctime()
    now = now.replace(' ', '_').replace(':', '-')
    writeProgressToLog = True
    logNmax = 7
    logFileName = './ProxDenois_' + str(logNmax) + '_' + now + '.log'
    outputFileName = './ProxDenois_' + str(logNmax) + '_' + now + '.json'
    main(logNmax, outputFileName, logFileName)
