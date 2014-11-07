import MySQLdb
import sys
import numpy as np
import matplotlib.pyplot as plt
from george import kernels
import george
import emcee
import triangle

db      = MySQLdb.connect(host='tddb.astro.washington.edu', user='tddb', passwd='tddb', db='Kepler')
cursor  = db.cursor()

def getKeplerData(kid):
    print "# Reading Database"
    sql = 'select TIME,PDCSAP_FLUX,PDCSAP_FLUX_ERR from source where KEPLERID = %s and LCFLAG = 1 and PDCSAP_FLUX>0 order by TIME' % (kid)
    cursor.execute(sql)
    results = cursor.fetchall()

    bjd  = np.array([x[0] for x in results])[1000:3000]
    flux = np.array([x[1] for x in results])[1000:3000]
    ferr = np.array([x[2] for x in results])[1000:3000]

    return bjd, flux, ferr

def residualPlot(x, y, dy, gp, title, fbase=12):
    fig, (sp1, sp2) = plt.subplots(2, sharex=True, figsize=(12,8))
    fig.subplots_adjust(hspace=0.1)

    # Top plot shows data and model
    xt = np.arange(min(x), max(x), 0.01)
    mut, covt = gp.predict(y, xt)
    std = np.sqrt(np.diag(covt))
    sp1.errorbar(x, y, yerr=dy, fmt="ro", alpha=0.25)
    sp1.plot(xt, mut, "k-")
    sp1.fill_between(xt, mut-std, mut+std, alpha=0.75, color="#4682b4")
    sp1.fill_between(xt, mut-3*std, mut+3*std, alpha=0.25, color="#4682b4")
    sp1.set_ylabel("Flux", weight="bold", fontsize=fbase+2)
    plt.setp(sp1.get_xticklabels()+sp1.get_yticklabels(), weight="bold", fontsize=fbase)

    # Bottom plot shows residuals from model
    mut, covt = gp.predict(y, x)
    # Now add the histogram of values to the standardized residuals plot
    resids = (y-mut) / np.sqrt(dy**2 + np.diag(covt))
    pdf, bin_edges = np.histogram(resids, bins=20)
    bin_edges = bin_edges[0:pdf.size]
    # Stretch the PDF so that it is readable
    pdf = pdf / float(pdf.max()) * 0.4 * (np.max(x) - np.min(x))
    sp2.barh(bin_edges, pdf, height=bin_edges[1]-bin_edges[0], left=np.min(x))
    expected_pdf = np.exp(-0.5 * bin_edges ** 2)
    expected_pdf = expected_pdf / expected_pdf.max() * 0.4 * (np.max(x) - np.min(x))
    sp2.plot(expected_pdf+np.min(x), bin_edges, 'DarkOrange', lw=2)
    sp2.plot(x, resids, "ro", alpha=0.25)
    sp2.axhline(y=0,  c='k', linestyle='--')
    sp2.set_xlabel("BJD", weight="bold", fontsize=fbase+2)
    sp2.set_ylabel("N Sigma", weight="bold", fontsize=fbase+2)
    plt.setp(sp2.get_xticklabels()+sp2.get_yticklabels(), weight="bold", fontsize=fbase)

    sp1.set_xlim(min(x), max(x))
    fig.suptitle(title, weight="bold", fontsize=fbase+4)

def modelGp(bjd, flux, ferr, ktype, lnamp, lnscale):
    x  = bjd
    y  = flux - np.mean(flux)
    dy = ferr

    k1 = np.exp(lnamp*2) * ktype(np.exp(lnscale)) 
    kernel = k1
    gp = george.GP(kernel, solver=george.HODLRSolver)
    gp.compute(x, dy)
    print "INITIAL GUESS LNLIKE", gp.lnlikelihood(y)
    residualPlot(x, y, dy, gp, "Initial Guess")


    # MASSIVE NOTE TO SELF:  the optimizer optimizes the log of the parameter vector.
    # To create a new GP object from the results we have to np.exp them.
    opt = gp.optimize(x, y, yerr=dy)
    params = np.exp(opt[0])
    k1 = params[0] * ktype(params[1])
    kopt = k1 
    gp = george.GP(kopt, solver=george.HODLRSolver)
    gp.compute(x, dy)
    print "AFTER OPT LNLIKE", gp.lnlikelihood(y)
    residualPlot(x, y, dy, gp, "Post Op")

    def lnlike(params, xl, yl, dyl):
        a, tau = np.exp(params)
        gp = george.GP(a * ktype(tau))
        gp.compute(xl, dyl)
        return gp.lnlikelihood(yl)

    ndim = len(params)
    nwalkers = 2 * ndim
    nburn = 100
    nstep = 500
    p0 = [np.array(np.log(params)) + 1e-8 * np.random.randn(ndim) for i in xrange(nwalkers)]
    sampler = emcee.EnsembleSampler(nwalkers, ndim, lnlike, args=(x,y,dy))
    print("Running burn-in...")
    p0, _, _ = sampler.run_mcmc(p0, nburn)
    sampler.reset()
    print("Running production...")
    sampler.run_mcmc(p0, nstep)
    figure = triangle.corner(sampler.flatchain, labels=["a", "tau"], truths=params, plot_contours=False)
    lnlike = sampler.flatlnprobability
    sortIdx  = np.argsort(lnlike)[::-1]
    sortPars = sampler.flatchain[sortIdx]
    bestPars = np.exp(sortPars[0])
    k1 = bestPars[0] * ktype(bestPars[1])
    kmcmc = k1 
    gp = george.GP(kmcmc, solver=george.HODLRSolver)
    gp.compute(x, dy)
    print "BEST MCMC STEP", gp.lnlikelihood(y)
    residualPlot(x, y, dy, gp, "Post Mcmc")

    opt = gp.optimize(x, y, yerr=dy)
    params = np.exp(opt[0])
    k1 = params[0] * ktype(params[1])
    kopt = k1 
    gp = george.GP(kopt, solver=george.HODLRSolver)
    gp.compute(x, dy)
    print "OPT POST MCMC", gp.lnlikelihood(y)
    residualPlot(x, y, dy, gp, "Opt Post Mcmc")

    import pdb; pdb.set_trace()


if __name__ == "__main__":
    kid = sys.argv[1]
    bjd, flux, ferr = getKeplerData(kid)

    modelGp(bjd, flux, ferr, kernels.ExpSquaredKernel, np.log(0.1), np.log(10))
    plt.errorbar(bjd, flux, yerr=ferr, fmt="ro")
    plt.show()