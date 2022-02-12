"""
This file includes the class of Synthetic data.
This class includes several methods 
    to generate several types of synthetic data.
    
In the GR paper we used "gen_correlated_noisy_features" method.
"""

class Synthetic_data:

    import sklearn.datasets as dt
    import numpy as np
    
    def __init__(self, samplesize, n_classes, sep_par, 
                 n_informative_features):
        self.n= samplesize
        self.n_class= n_classes
        self.delta= sep_par
        self.n_inf= n_informative_features
    
    def gen_xor_blobs_circ(self, cluster_std= 1, n_noisy_features= 0):
        import numpy as np
        import sklearn.datasets as dt
        
        centers= np.array(self.get_means_bimodal())
        x, y = dt.make_blobs(n_samples=self.n, n_features=self.n_inf,
            cluster_std= cluster_std, centers= centers, shuffle=False)
        
        if n_noisy_features != 0:
            x_noise,  y_noise= dt.make_blobs(n_samples=self.n, n_features=n_noisy_features,
                                            cluster_std= np.mean(cluster_std), 
                                             centers= 1, center_box=(-1.0, 1.0))    
            x= np.concatenate([x, x_noise], axis= 1)
        
        for i in range(self.n_class):
            y[y==(self.n_class+i)] = i
        return x, y

    def gen_blobs_circ(self, cluster_std= 1, n_noisy_features= 0):
        import numpy as np
        import sklearn.datasets as dt
        
        centers= np.array(self.get_means_unimodal())
        x, y = dt.make_blobs(n_samples=self.n, n_features=self.n_inf,
            cluster_std= cluster_std, centers= centers, shuffle=False)
        
        if n_noisy_features != 0:
            x_noise,  y_noise= dt.make_blobs(n_samples=self.n, n_features=n_noisy_features,
                                            cluster_std= np.mean(cluster_std), 
                                             centers= 1, center_box=(-1.0, 1.0))    
            x= np.concatenate([x, x_noise], axis= 1)
        
        return x, y
        
    
    def gen_correlated_noisy_features(self, 
                                      n_blocks, block_sizes, 
                                      block_group_vars, block_corr, 
                                      n_noisy_features= 0,
                                      bimodal= False):
        """
        Generates Gaussian data with a blocked covariance matrix

        Arguments:
            n_blocks (int):   number of blocks
            block_sizes (array): size of each block
            block_group_vars (list of arrays with length n_classes): each array includes variance in each block
            block_corr (lsit of arrays with length n_classes): each array contains correlation in each block
        Returns:
            X: data matrix of shape [n, sum(block_sizes)]
        """
        import numpy as np
        import sklearn.datasets as dt
        from numpy.random import multivariate_normal as mvn
        from numpy.random import multinomial
        
        def gen_x_perClass(n, m, k, l, s, r):
            """
            Generates Gaussian data with a blocked covariance matrix

            Arguments:
                n (int):   sample size
                m (array): mean vector
                k (int):   number of blocks
                l (array): size of each block
                s (array): variance in each block
                r (array): correlation in each block
            Returns:
                X: data matrix of size n x (l[0]+...+l[k-1]) 
            """
            X = np.zeros([n,sum(l)])
            t = 0
            for i in range(k):
                cov = s[i]*((1-r[i])*np.identity(l[i])+r[i]*np.ones([l[i],l[i]]))
                X[:,t:t+l[i]] = mvn(m[t:t+l[i]],cov, n)
                t = t + l[i]
            return X
        
        
        c = [1.0/self.n_class] * self.n_class
        class_size = np.random.multinomial(self.n, c)
        
        x= np.empty([1, sum(block_sizes)])
        y= np.array([0]) 
        
        if bimodal is False:
            m_= self.get_means_unimodal()
            for i in range(self.n_class):
                mi= m_[i,:]
    
                xi= gen_x_perClass(class_size[i], mi, 
                                    n_blocks, block_sizes, block_group_vars[i], 
                                    block_corr[i])
                yi= np.array([i]*class_size[i])
                x= np.concatenate([x, xi])
                y= np.concatenate([y, yi])
            if n_noisy_features != 0:
                x_noise,  y_noise= dt.make_blobs(n_samples=self.n, n_features=n_noisy_features,
                                        cluster_std= np.mean(block_group_vars), 
                                        centers= np.array([0]*n_noisy_features).reshape(1,-1))
                                          # centers= np.mean(m_, axis= 0).reshape([1, m_.shape[1]]))
                return np.concatenate([x[1:, :], x_noise], axis= 1), y[1:]
            else:
                return  x[1:, :], y[1:]                
            
        elif bimodal is True:
            m_1, m_2= self.get_means_bimodal(n_noisy_features= n_noisy_features)          
            for i in range(self.n_class):
                mi_1= m_1[i]#[i,:] 
                mi_2= m_2[i]                                
                ni= int(class_size[i])
                xi_1= gen_x_perClass(int(ni/2), mi_1, 
                                   n_blocks, block_sizes, block_group_vars[i], 
                                   block_corr[i])
                
                xi_2= gen_x_perClass(ni-int(ni/2), mi_2, 
                                   n_blocks, block_sizes, block_group_vars[i], 
                                   block_corr[i])
                
                xi= np.concatenate([xi_1, xi_2])
                yi= np.array([i]*class_size[i])
                x= np.concatenate([x, xi])
                y= np.concatenate([y, yi])
            if n_noisy_features != 0:
                x_noise,  y_noise= dt.make_blobs(n_samples=self.n, n_features=n_noisy_features,
                                        cluster_std= np.mean(block_group_vars), 
                                        centers= (np.mean(mi_1+mi_2)*np.ones(n_noisy_features).reshape(1,n_noisy_features)))
                    
                return  np.concatenate([x[1:, :], x_noise], axis= 1), y[1:]
            else:
                return x[1:, :], y[1:]

#-------------------------------------------------------------------------------------------
    def get_means_bimodal(self, n_noisy_features):
        import numpy as np
        from math import sin, cos
        pi= np.pi
        theta= pi/self.n_class
        centers1= {}
        centers2= {}
        for i in range(self.n_class):   
            cent1= [self.delta*sin(i*theta), self.delta*cos(i*theta)]
            cent2= [-self.delta*sin(i*theta), -self.delta*cos(i*theta)]
            centers1[i]= (int(self.n_inf/2) * cent1 + cent1[:self.n_inf%2])    
            centers2[i]= (int(self.n_inf/2) * cent2 + cent2[:self.n_inf%2])
        return centers1, centers2
    
    def get_means_unimodal(self):
        import numpy as np
        from math import sin, cos
        pi= np.pi
        theta= pi/4
        centers= []
        for i in range(self.n_class):   
            cent= [ self.delta * cos(theta + 2*pi*i/self.n_class), 
                    self.delta * sin(theta + 2*pi*i/self.n_class)]  
            centers= centers+ [(int(self.n_inf/2) * cent + cent[:self.n_inf%2])]
        return np.array(centers) 

    
