# Functions
def plotClusters(thetas,z,samples):
    thetameans = []
    K = len(thetas)
    plt.figure(figsize=(fig_len,fig_wid))
    ax=plt.gca()
    ax.set_facecolor('white')
    ax.tick_params(labelsize=25)
    ax.set_facecolor('white')
    ax.grid(color='k', linestyle='-.', linewidth=0.3)

    for k in range(K):
        thetameans.append(thetas[k][0])
    thetameans = np.array(thetameans)
    for k in range(K):
        plt.scatter(samples[z == k,0],samples[z == k,1],marker='*',s=m_size)
    plt.legend([str(k) for k in range(K)])
    #plt.scatter(thetameans[:,0],thetameans[:,1],marker='x')
    for k in range(K):
        plt.text(thetameans[k,0],thetameans[k,1],str(k))
def multivariatet(mu,Sigma,N,M):
    '''
    Output:
    Produce M samples of d-dimensional multivariate t distribution
    Input:
    mu = mean (d dimensional numpy array or scalar)
    Sigma = scale matrix (dxd numpy array)
    N = degrees of freedom
    M = # of samples to produce
    '''
    d = len(Sigma)
    g = np.tile(np.random.gamma(N/2.,2./N,M),(d,1)).T
    Z = np.random.multivariate_normal(np.zeros(d),Sigma,M)
    return mu + Z/np.sqrt(g)
def normalinvwishartsample(params):
    '''
    Generate sample from a Normal Inverse Wishart distribution

    Inputs:
    params - Parameters for the NIW distribution 
        mu    - Mean parameter: n x 1 numpy array
        W     - Precision parameter: d x d numpy array
        kappa - Scalar parameter for normal distribution covariance matrix
        nu    - Scalar parameter for Wishart distribution

    Output:
    Sample - Sample mean vector, mu_s and Sample covariance matrix, W_s
    '''
    mu,W,kappa,nu = params
    # first sample W from a Inverse Wishart distribution
    W_s = invwishart(df=nu, scale=W).rvs()
    mu_s = np.random.multivariate_normal(mu.flatten(),W_s/kappa,1) 
    return np.transpose(mu_s),W_s
def normalinvwishartmarginal(X,params):
    '''
    Marginal likelihood of dataset X using a Normal Inverse Wishart prior

    Inputs:
    X      - Dataset matrix: n x d numpy array
    params - Parameters for the NIW distribution 
        mu    - Mean parameter: n x 1 numpy array
        W     - Precision parameter: d x d numpy array
        kappa - Scalar parameter for normal distribution covariance matrix
        nu    - Scalar parameter for Wishart distribution

    Output:
    Marginal likelihood of X - scalar
    '''
    mu,W,kappa,nu = params
    mu=X

    n = X.shape[0]
    d = X.shape[1]
    nu_n = nu + n
    kappa_n = kappa + n
    X_mean = np.mean(X,axis=0)
    X_mean = X_mean[:,np.newaxis]
    S = scatter(X)
    W_n = W + S + ((kappa*n)/(kappa+n))*np.dot(mu - X_mean,np.transpose(mu - X_mean))
    #(1/np.power(np.pi,n*d*0.5))*(gamma(nu_n*0.5)/gamma(nu*0.5))*(np.power(np.linalg.det(W),nu*0.5)/np.power(np.linalg.det(W_n),nu_n*0.5))*np.power(kappa/kappa_n,0.5*d)
    return (1/np.power(np.pi,n*d*0.5))*(gamma(nu_n*0.5)/gamma(nu*0.5))*(np.power(np.linalg.det(W)/np.linalg.det(W_n),nu*0.5)/np.power(np.linalg.det(W_n),(nu_n-nu)*0.5))*np.power(kappa/kappa_n,0.5*d)
def scatter(x):
    return np.dot(np.transpose(x - np.mean(x,0)),x - np.mean(x,0))
def plotAnomalies(I,samples):
    for k in range(2):
        plt.figure(figsize=(fig_len,fig_wid))
        ax=plt.gca()
        ax.set_facecolor('white')
        ax.tick_params(labelsize=20)
        ax.set_facecolor('white')
        ax.grid(color='k', linestyle='-.', linewidth=0.3)
        plt.scatter(samples[I == k,0],samples[I == k,1],marker='*',s=m_size)
def kmeans__(data,k,l,maxiters=100,eps=0.0001):

    # select k cluster centers
    C = data[np.random.permutation(range(data.shape[0]))[0:k],:]
    objVal = 0
    for jj in range(maxiters):
        # compute distance of each point to the clusters
        dMat = pdist2(data,C)
        d = np.min(dMat,axis=1).flatten()
        c = np.argmin(dMat,axis=1).flatten()
        # sort points by distance to their closest center
        inds = np.argsort(d)[::-1]
        linds = inds[0:l]
        cinds = inds[l+1:]
        # extract the non-outlier data objects
        ci = c[cinds]
        # recompute the means
        for kk in range(k):
            C[kk,:] = np.mean(data[np.where(ci == kk)[0],:],axis=0)
        # compute objective function
        objVal_ = objVal
        objVal = 0
        for kk in range(k):
            objVal += np.sum(pdist2(data[np.where(ci == kk)[0],:],C[kk,:]))
        if np.abs(objVal - objVal_) < eps:
            break
    # one final time
    dMat = pdist2(data,C)
    c = np.argmin(dMat,axis=1).flatten()
    return linds, C, c
def pdist2(X,C):
    if len(C.shape) == 1:
        C = C[:,np.newaxis]
    distMat = np.zeros([X.shape[0],C.shape[0]])
    for i in range(X.shape[0]):
        for j in range(C.shape[0]):
            distMat[i,j] = np.linalg.norm(X[i,:] - C[j,:])
    return distMat
def precAtK(true,predicted):
    # find number of anomalies
    k = np.sum(true)
#     print("k=",k)
    # find the score of the k^th predicted anomaly
    v = np.sort(predicted,axis=0)[::-1][k-1]
#     print("v=",v)
    # find all objects that are above the threshold
    inds = np.where(predicted >= v)[0]
#     print("inds=",inds)
#     print("np.sum(true[inds])=",np.sum(true[inds]))
#     print("len(inds)=",len(inds))
#     print("np.sum(true[inds])/len(inds)=",np.sum(true[inds])/len(inds))
    return float(np.sum(true[inds]))/float(len(inds))
def averageRank(true,predicted):
    inds = np.where(true == 1)[0]
    s = np.argsort(predicted)[::-1]
    v = []
    for ind in inds:
        v.append(float(np.where(s == ind)[0]+1))
    return np.mean(v)
def purity_score(y_true, y_pred):
    """Purity score

    To compute purity, each cluster is assigned to the class which is most frequent 
    in the cluster [1], and then the accuracy of this assignment is measured by counting 
    the number of correctly assigned documents and dividing by the number of documents.
    We suppose here that the ground truth labels are integers, the same with the predicted clusters i.e
    the clusters index.

    Args:
        y_true(np.ndarray): n*1 matrix Ground truth labels
        y_pred(np.ndarray): n*1 matrix Predicted clusters
    
    Returns:
        float: Purity score
    
    References:
        [1] https://nlp.stanford.edu/IR-book/html/htmledition/evaluation-of-clustering-1.html
    """
    # matrix which will hold the majority-voted labels
    y_voted_labels = np.zeros(y_true.shape)
    # Ordering labels
    ## Labels might be missing e.g with set like 0,2 where 1 is missing
    ## First find the unique labels, then map the labels to an ordered set
    ## 0,2 should become 0,1
    labels = np.unique(y_true)
    ordered_labels = np.arange(labels.shape[0])
    for k in range(labels.shape[0]):
        y_true[y_true==labels[k]] = ordered_labels[k]
    # Update unique labels
    labels = np.unique(y_true)
    # We set the number of bins to be n_classes+2 so that 
    # we count the actual occurence of classes between two consecutive bin
    # the bigger being excluded [bin_i, bin_i+1[
    bins = np.concatenate((labels, [np.max(labels)+1]), axis=0)

    for cluster in np.unique(y_pred):
        hist, _ = np.histogram(y_true[y_pred==cluster], bins=bins)
        # Find the most present label in the cluster
        winner = np.argmax(hist)
        y_voted_labels[y_pred==cluster] = winner
    
    return accuracy_score(y_true, y_voted_labels)

def _estimate_log_gaussian_prob(X, means, precisions_chol, covariance_type):
    """Estimate the log Gaussian probability.
    Parameters
    ----------
    X : array-like, shape (n_samples, n_features)
    means : array-like, shape (n_components, n_features)
    precisions_chol : array-like
        Cholesky decompositions of the precision matrices.
        'full' : shape of (n_components, n_features, n_features)
        'tied' : shape of (n_features, n_features)
        'diag' : shape of (n_components, n_features)
        'spherical' : shape of (n_components,)
    covariance_type : {'full', 'tied', 'diag', 'spherical'}
    Returns
    -------
    log_prob : array, shape (n_samples, n_components)
    """
    n_samples, n_features = X.shape
    n_components, _ = means.shape
    # det(precision_chol) is half of det(precision)
    log_det = _compute_log_det_cholesky(
        precisions_chol, covariance_type, n_features)

    if covariance_type == 'full':
        log_prob = np.empty((n_samples, n_components))
        for k, (mu, prec_chol) in enumerate(zip(means, precisions_chol)):
            y = np.dot(X, prec_chol) - np.dot(mu, prec_chol)
            log_prob[:, k] = np.sum(np.square(y), axis=1)

    elif covariance_type == 'tied':
        log_prob = np.empty((n_samples, n_components))
        for k, mu in enumerate(means):
            y = np.dot(X, precisions_chol) - np.dot(mu, precisions_chol)
            log_prob[:, k] = np.sum(np.square(y), axis=1)

    elif covariance_type == 'diag':
        precisions = precisions_chol ** 2
        log_prob = (np.sum((means ** 2 * precisions), 1) -
                    2. * np.dot(X, (means * precisions).T) +
                    np.dot(X ** 2, precisions.T))

    elif covariance_type == 'spherical':
        precisions = precisions_chol ** 2
        log_prob = (np.sum(means ** 2, 1) * precisions -
                    2 * np.dot(X, means.T * precisions) +
                    np.outer(row_norms(X, squared=True), precisions))
    return -.5 * (n_features * np.log(2 * np.pi) + log_prob) + log_det
def _compute_precision_cholesky(covariances, covariance_type):
    """Compute the Cholesky decomposition of the precisions.
    Parameters
    ----------
    covariances : array-like
        The covariance matrix of the current components.
        The shape depends of the covariance_type.
    covariance_type : {'full', 'tied', 'diag', 'spherical'}
        The type of precision matrices.
    Returns
    -------
    precisions_cholesky : array-like
        The cholesky decomposition of sample precisions of the current
        components. The shape depends of the covariance_type.
    """
    estimate_precision_error_message = (
        "Fitting the mixture model failed because some components have "
        "ill-defined empirical covariance (for instance caused by singleton "
        "or collapsed samples). Try to decrease the number of components, "
        "or increase reg_covar.")

    if covariance_type == 'full':
        n_components, n_features, _ = covariances.shape
        precisions_chol = np.empty((n_components, n_features, n_features))
        for k, covariance in enumerate(covariances):
            try:
                cov_chol = linalg.cholesky(covariance, lower=True)
            except linalg.LinAlgError:
                raise ValueError(estimate_precision_error_message)
            precisions_chol[k] = linalg.solve_triangular(cov_chol,
                                                         np.eye(n_features),
                                                         lower=True).T
    elif covariance_type == 'tied':
        _, n_features = covariances.shape
        try:
            cov_chol = linalg.cholesky(covariances, lower=True)
        except linalg.LinAlgError:
            raise ValueError(estimate_precision_error_message)
        precisions_chol = linalg.solve_triangular(cov_chol, np.eye(n_features),
                                                  lower=True).T
    else:
        if np.any(np.less_equal(covariances, 0.0)):
            raise ValueError(estimate_precision_error_message)
        precisions_chol = 1. / np.sqrt(covariances)
    return precisions_chol
def _estimate_log_prob(means_,precisions_cholesky_,covariance_type,degrees_of_freedom_,mean_precision_, X):
        _, n_features = X.shape
        # We remove `n_features * np.log(degrees_of_freedom_)` because
        # the precision matrix is normalized
        log_gauss = (_estimate_log_gaussian_prob(
            X, means_, precisions_cholesky_, covariance_type) -
            .5 * n_features * np.log(degrees_of_freedom_))

        log_lambda = n_features * np.log(2.) + np.sum(digamma(
            .5 * (degrees_of_freedom_ -
                  np.arange(0, n_features)[:, np.newaxis])), 0)

        return log_gauss + .5 * (log_lambda -
                                 n_features / mean_precision_)
def _compute_log_det_cholesky(matrix_chol, covariance_type, n_features):
    """Compute the log-det of the cholesky decomposition of matrices.
    Parameters
    ----------
    matrix_chol : array-like
        Cholesky decompositions of the matrices.
        'full' : shape of (n_components, n_features, n_features)
        'tied' : shape of (n_features, n_features)
        'diag' : shape of (n_components, n_features)
        'spherical' : shape of (n_components,)
    covariance_type : {'full', 'tied', 'diag', 'spherical'}
    n_features : int
        Number of features.
    Returns
    -------
    log_det_precision_chol : array-like, shape (n_components,)
        The determinant of the precision matrix for each component.
    """
    if covariance_type == 'full':
        n_components, _, _ = matrix_chol.shape
        log_det_chol = (np.sum(np.log(
            matrix_chol.reshape(
                n_components, -1)[:, ::n_features + 1]), 1))

    elif covariance_type == 'tied':
        log_det_chol = (np.sum(np.log(np.diag(matrix_chol))))

    elif covariance_type == 'diag':
        log_det_chol = (np.sum(np.log(matrix_chol), axis=1))

    else:
        log_det_chol = n_features * (np.log(matrix_chol))

    return log_det_chol
def _estimate_gaussian_parameters(X, resp, reg_covar, covariance_type):
    """Estimate the Gaussian distribution parameters.
    Parameters
    ----------
    X : array-like, shape (n_samples, n_features)
        The input data array.
    resp : array-like, shape (n_samples, n_components)
        The responsibilities for each data sample in X.
    reg_covar : float
        The regularization added to the diagonal of the covariance matrices.
    covariance_type : {'full', 'tied', 'diag', 'spherical'}
        The type of precision matrices.
    Returns
    -------
    nk : array-like, shape (n_components,)
        The numbers of data samples in the current components.
    means : array-like, shape (n_components, n_features)
        The centers of the current components.
    covariances : array-like
        The covariance matrix of the current components.
        The shape depends of the covariance_type.
    """
    nk = resp.sum(axis=0) + 10 * np.finfo(resp.dtype).eps
    means = np.dot(resp.T, X) / nk[:, np.newaxis]
    covariances = {"full": _estimate_gaussian_covariances_full                   
                  }[covariance_type](resp, X, nk, means, reg_covar)
    return nk, means, covariances
def _estimate_gaussian_covariances_full(resp, X, nk, means, reg_covar):
    """Estimate the full covariance matrices.
    Parameters
    ----------
    resp : array-like, shape (n_samples, n_components)
    X : array-like, shape (n_samples, n_features)
    nk : array-like, shape (n_components,)
    means : array-like, shape (n_components, n_features)
    reg_covar : float
    Returns
    -------
    covariances : array, shape (n_components, n_features, n_features)
        The covariance matrix of the current components.
    """
    n_components, n_features = means.shape
    covariances = np.empty((n_components, n_features, n_features))
    for k in range(n_components):
        diff = X - means[k]
        covariances[k] = np.dot(resp[:, k] * diff.T, diff) / nk[k]
        covariances[k].flat[::n_features + 1] += reg_covar
    return covariances
def initialization(X,K,numiters,r0,alpha=1):
    ########################### Data preprocessing
    X,z,thetas,N,params,D=preprocess(X,K)

    clusters, sizes = np.unique(z, return_counts=True)
    m_para=sizes/N
    F=np.zeros(N)
#     params=tuple((np.array(pd.DataFrame(X).mean()),((np.array(pd.DataFrame(X).cov()))), 1, D))
    
#     m_para,F=F_est(np.ones,N,N,thetas,params,X)
    
    threshold=0.05
    
    I=(np.random.binomial(1, threshold, N))
        
    return X,z,I,thetas,N,params,D,clusters,sizes,m_para,F,threshold
def convergence_check(thetas,centroids_old,conv_criteria):
    centroids=np.copy(np.array(list([thetas[i][0] for i in range(len(thetas))])))
    if len(centroids)<len(centroids_old):
        change=np.sum(list(np.min(np.abs(np.linalg.norm(centroids_old-centroids[i],axis=1))) for i in range(len(centroids))))
    else:
        change=np.sum(list(np.min(np.abs(np.linalg.norm(centroids_old[i]-centroids,axis=1))) for i in range(len(centroids_old))))
    return change
def preprocess(X,K):
    
# Try different mean precision prior ie params[2] : No difference
# mean_precision_prior float | None, optional.
# The precision prior on the mean distribution (Gaussian). Controls the extent of where means can be placed. 
# Larger values concentrate the cluster means around mean_prior. The value of the parameter must be greater 
# than 0. If it is None, it is set to 1.

# Try different reg_covar : Too volatile
    if type(X) == list:
        X = np.array(X)
    if len(X.shape) == 1:
        X = X[:,np.newaxis]
    X=np.array(X).astype(float)

    N = X.shape[0] #rows: observations
    D = X.shape[1] #columns: dimensions

    # Fit your data on the scaler object
    scaler = StandardScaler()
    X = scaler.fit_transform(X)
    
#     X=normalize(X)
    # Initialize z
#     z=np.random.randint(K,size=N)
    z=KMeans(K).fit(X).predict(X)
    z+=1

    if D>1:
        params = tuple((np.mean(X,axis=0),(np.cov(X.T)), 1, D))
    elif D==1:
        params = tuple((np.mean(X),(np.var(X.T)),1, D))

    z=np.random.randint(K,size=N)

    thetas=[normalinvwishartsample(params) for k in range(K)]

    return X,z,thetas,N,params,D
def remove_cluster_new(X,z,K,thetas,params):
#     if len(thetas)>len(np.unique(np.abs(z))):
#         print(len(thetas)-len(np.unique(np.abs(z)))," clusters removed", len(np.unique(np.abs(z))) )
    N=len(z)
    z_temp=np.copy(z)
    clusters, sizes = np.unique(np.abs(z_temp), return_counts=True)

    c2=pd.DataFrame(clusters).copy()
    temp=c2.index.copy()+1
    c2.index=c2[0].copy()
    c2[0]=temp.copy()
    z=np.multiply(np.copy(c2[0][np.abs(z_temp)]),np.sign(z_temp+0.5))
    
    clusters, sizes = np.unique(np.abs(z), return_counts=True)
    K=len(clusters)
    
    return z,K,thetas
def compute_mixture_pdf(means_,precisions_cholesky_,covariance_type,mean_precision_, X,N,sizes):
    degrees_of_freedom_=sizes+X.shape[1]
    log_probs=_estimate_log_prob(means_,precisions_cholesky_,covariance_type,degrees_of_freedom_,mean_precision_, X)
    MN=(np.exp(log_probs))
    F=np.dot(sizes/N,np.exp(log_probs).T)
    return degrees_of_freedom_,log_probs,MN,F
def compute_cluster_params(z,X,params,clusters,sizes,ind_matrix,reg_covar,covariance_type):
    K=(len(clusters))
    N=len(z)
    thetas=[]
    for k in (clusters-1):
        ind_k=np.where((z) == (k+1))[0]
        c = len(ind_k)
        if c<1:
#             print("Group anomaly")
            ind_k=np.where(np.abs(z) == (k+1))[0]
            c = len(ind_k)
        thetas.append((_estimate_gaussian_parameters(X[ind_k], 
                                            np.ones((c,1)), reg_covar, covariance_type)[1:3]))    
    nk=sizes
    means_=np.array([thetas[k][0].T for k in clusters-1])[:,:,0]
    covariances=np.array([thetas[k][1][0,:,:] for k in clusters-1])
    para_tuple=nk,means_,covariances
    precisions_cholesky_= np.array([_compute_precision_cholesky(cov, 
                                                covariance_type) for cov in [covariances]])[0,:,:,:]
    
    return para_tuple,thetas,nk,means_,covariances,precisions_cholesky_
def _log_wishart_norm(degrees_of_freedom, log_det_precisions_chol, n_features):
    """Compute the log of the Wishart distribution normalization term.
    Parameters
    ----------
    degrees_of_freedom : array-like, shape (n_components,)
        The number of degrees of freedom on the covariance Wishart
        distributions.
    log_det_precision_chol : array-like, shape (n_components,)
         The determinant of the precision matrix for each component.
    n_features : int
        The number of features.
    Return
    ------
    log_wishart_norm : array-like, shape (n_components,)
        The log normalization of the Wishart distribution.
    """
    # To simplify the computation we have removed the np.log(np.pi) term
    return -(degrees_of_freedom * log_det_precisions_chol +
             degrees_of_freedom * n_features * .5 * math.log(2.) +
             np.sum(gammaln(.5 * (degrees_of_freedom -
                                  np.arange(n_features)[:, np.newaxis])), 0))
def compute_log_likelihhod(z,sizes,K,precisions_cholesky_,covariance_type,features,degrees_of_freedom_,
                           mean_precision_):
        # Contrary to the original formula, we have done some simplification
        # and removed all the constant terms.
        log_resp=np.log(np.abs(z))
        weight_concentration_ = (
                1. + sizes,
                (1/K +
                 np.hstack((np.cumsum(sizes[::-1])[-2::-1], 0))))

        # We removed `.5 * features * np.log(degrees_of_freedom_)`
        # because the precision matrix is normalized.
        log_det_precisions_chol = (_compute_log_det_cholesky(
            precisions_cholesky_, covariance_type, features) -
            .5 * features * np.log(degrees_of_freedom_))

        log_wishart = np.sum(_log_wishart_norm(
            degrees_of_freedom_, log_det_precisions_chol, features))
        
        log_norm_weight = -np.sum(betaln(weight_concentration_[0],
                                         weight_concentration_[1]))

        curr_log_likelihood=(-np.sum(np.exp(log_resp) * log_resp) -
                log_wishart - log_norm_weight -
                0.5 * features * np.sum(np.log(mean_precision_)))
        return curr_log_likelihood
def ppsa_vals(F,I,threshold):
    N=len(F)
    u=np.unique(F)
    ps1=F
    domain=((u>np.quantile(F,0.01))*1==(u<np.quantile(F,0.3))*1)
    G_Y_domain=u*domain
    G_Y_domain=(G_Y_domain[G_Y_domain>0])

    G_Y=domain*[np.sum(F[(F<=u)]) for n,u in enumerate(np.unique(F))]
    G_Y=G_Y[G_Y>0]
    if len(G_Y)!=0:
        G_Y=np.array(G_Y/max(G_Y))

        g_Y=np.diff(G_Y)/np.diff(G_Y_domain)

        aa=np.array(list(stats.percentileofscore(F, i)/100 for i in np.unique(F)))
        th3=aa[aa<0.3][np.argmax(np.abs(np.diff(np.quantile(G_Y,aa[aa<0.3]))))]
    else:
        th3=threshold
    len_tail=np.int(th3*N) #drop in F

    u=np.quantile(ps1, th3)
    inds = np.where(ps1<=u)[0]

    iz=np.union1d(inds, np.where(I==1)[0])
#     inds0=iz[np.argsort(ps1[iz])[:min(np.int(0.3*N),len(iz),np.int(th3*N))]]
    inds0=np.argsort(ps1)[:min(np.int(0.3*N),len(iz),np.int(th3*N))]
    psa = np.abs(ps1[inds0] - u) 

    gpdparams = stats.genpareto.fit(psa)
    i_ppsa=np.zeros(N)
    
    i_ppsa[inds0] = stats.genpareto(1,0,gpdparams[2]).cdf(psa)
    ppsa=np.ones(N)
    
    ppsa[inds0]=1-(i_ppsa[inds0])
    
#     i_ppsa[iz] = stats.genpareto(1,0,gpdparams[2]).cdf(np.abs(ps1[iz] - u))
#     ppsa=np.ones(N)
#     ppsa[iz]=1-(i_ppsa[iz])
    
    ppsa[ppsa>1]=1
    ppsa[ppsa==0]=sys.float_info.min
    return ppsa,i_ppsa, inds, inds0,u,F, threshold

