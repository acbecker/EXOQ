import sys
import numpy as np
import kplr
client = kplr.API()

if __name__ == "__main__":
    KOI  = sys.argv[1]
    koic = client.koi(KOI)
    pfiles = koic.get_target_pixel_files(fetch=True, clobber=False)
    for pfile in pfiles:
        with pfile.open() as f:
            data = f[1].data
            mask = f[2].data
            idx  = np.where(mask == 3)
            time = data["time"]
            flux = data["flux"]
            for i, (t,f) in enumerate(zip(time, flux)):
                coeffs = f[idx]/np.sum(f[idx])
                print t, coeffs
