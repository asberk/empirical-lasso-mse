import pdmse
import sys
import jsonWrite


def main(logNmax, filename, logfile=sys.stdout):
    pdmse_dict = {'LS_x_norm1':
                  pdmse.pdmse_batch(logNmax, func=pdmse.pdmse_golden_tau,
                                    verbose=logfile),
                  'BP_z_norm2sq':
                  pdmse.pdmse_batch(logNmax, theta='sqNormZ',
                                    verbose=logfile)}
    fp = open(logfile, 'a', encoding='utf-8')
    print('\nRuns complete. Saving dict to json...', file=fp, end='')
    jsonWrite.dict(filename, pdmse_dict)
    print('complete!', file=fp)
    fp.close()
    return

if __name__ == "__main__":
    from datetime import datetime as dt
    now = dt.now().ctime()
    now = now.replace(' ', '_').replace(':', '-')
    writeProgressToLog = True
    logNmax = 7
    logFileName = './ProxDenois_LSBP_' + str(logNmax) + '_' + now + '.log'
    outputFileName = './ProxDenois_LSBP_' + str(logNmax) + '_' + now + '.json'
    main(logNmax, outputFileName, logFileName)
