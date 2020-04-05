import splitdataset
import lbp3

age_params = [[12, 50, 100],
              [15, 30, 100],
              [10, 30, 100]]

perc = 0.70
divisions = [3, 8, 9]
classifiers = ['/haarcascade_frontalface_alt.xml', '/haarcascade_frontalface_alt2.xml', '/haarcascade_frontalface_default.xml']
for ap in age_params:

    splitdataset.split(ap[0], ap[1], 'age', perc)

    for c in classifiers:
        for d in divisions:
            print('CONFIGURAZIONE: ' + str(ap) + '\nDIVISIONI: ' + str(d) + '\nCLASSIFIER: ' + c)
            l, t, a, g = lbp3.lbp3_fun(ap, d, c)
            print('\n' + l + t + '\n' + l + a + '\n' + l)
