
class Spec_Clust_unorm:
    def __init__(self, min_num_spkrs=1, max_num_spkrs=10):

        self.min_num_spkrs = min_num_spkrs
        self.max_num_spkrs = max_num_spkrs

    def do_spec_clust(self, X, k_oracle, p_val):
        """Function for spectral clustering.
        Arguments
        ---------
        X : array
            (n_samples, n_features).
            Embeddings extracted from the model.
        k_oracle : int
            Number of speakers (when oracle number of speakers).
        p_val : float
            p percent value to prune the affinity matrix.
        """

        # Similarity matrix computation
        sim_mat = self.get_sim_mat(X)

        # Refining similarity matrix with p_val
        prunned_sim_mat = self.p_pruning(sim_mat, p_val)

        # Symmetrization
        sym_prund_sim_mat = 0.5 * (prunned_sim_mat + prunned_sim_mat.T)

        # Laplacian calculation
        laplacian = self.get_laplacian(sym_prund_sim_mat)

        # Get Spectral Embeddings
        emb, num_of_spk = self.get_spec_embs(laplacian, k_oracle)

        # Perform clustering
        # import pdb;pdb.set_trace()
        self.cluster_embs(emb, num_of_spk)
    def get_sim_mat(self, X):
        """Returns the similarity matrix based on cosine similarities.
        Arguments
        ---------
        X : array
            (n_samples, n_features).
            Embeddings extracted from the model.
        Returns
        -------
        M : array
            (n_samples, n_samples).
            Similarity matrix with cosine similarities between each pair of embedding.
        """

        # Cosine similarities
        M = cosine_similarity(X, X)
        return M

    def p_pruning(self, A, pval):
        """Refine the affinity matrix by zeroing less similar values.
        Arguments
        ---------
        A : array
            (n_samples, n_samples).
            Affinity matrix.
        pval : float
            p-value to be retained in each row of the affinity matrix.
        Returns
        -------
        A : array
            (n_samples, n_samples).
            Prunned affinity matrix based on p_val.
        """

        n_elems = int((1 - pval) * A.shape[0])

        # For each row in a affinity matrix
        for i in range(A.shape[0]):
            low_indexes = np.argsort(A[i, :])
            low_indexes = low_indexes[0:n_elems]

            # Replace smaller similarity values by 0s
            A[i, low_indexes] = 0

        return A
    def get_laplacian(self, M):
        """Returns the un-normalized laplacian for the given affinity matrix.
        Arguments
        ---------
        M : array
            (n_samples, n_samples)
            Affinity matrix.
        Returns
        -------
        L : array
            (n_samples, n_samples)
            Laplacian matrix.
        """

        M[np.diag_indices(M.shape[0])] = 0
        D = np.sum(np.abs(M), axis=1)
        D = np.diag(D)
        # L is a signed laplacian given the definition of D
        L = D - M
        return L
    def get_spec_embs(self, L, k_oracle):
        """Returns spectral embeddings and estimates the number of speakers
        using maximum Eigen gap.
        Arguments
        ---------
        L : array (n_samples, n_samples)
            Laplacian matrix.
        k_oracle : int
            Number of speakers when the condition is oracle number of speakers,
            else None.
        Returns
        -------
        emb : array (n_samples, n_components)
            Spectral embedding for each sample with n Eigen components.
        num_of_spk : int
            Estimated number of speakers. If the condition is set to the oracle
            number of speakers then returns k_oracle.
        """

        lambdas, eig_vecs = scipy.linalg.eigh(L)

        # if params["oracle_n_spkrs"] is True:
        if k_oracle is not None:
            num_of_spk = k_oracle
        else:
            lambda_gap_list = self.getEigenGaps(lambdas[0 : self.max_num_spkrs])

            num_of_spk = (
                np.argmax(
                    lambda_gap_list[: min(self.max_num_spkrs, len(lambda_gap_list))]
                )
                + 1
            )

            if num_of_spk < self.min_num_spkrs:
                num_of_spk = self.min_num_spkrs

        emb = eig_vecs[:, 0:num_of_spk]

        return emb, num_of_spk
    def cluster_embs(self, emb, k):
        """Clusters the embeddings using kmeans.
        Arguments
        ---------
        emb : array (n_samples, n_components)
            Spectral embedding for each sample with n Eigen components.
        k : int
            Number of clusters to kmeans.
        Returns
        -------
        self.labels_ : self
            Labels for each sample embedding.
        """
        _, self.labels_, _ = k_means(emb, k, random_state=1)

    def getEigenGaps(self, eig_vals):
        """Returns the difference (gaps) between the Eigen values.
        Arguments
        ---------
        eig_vals : list
            List of eigen values
        Returns
        -------
        eig_vals_gap_list : list
            List of differences (gaps) between adjancent Eigen values.
        """

        eig_vals_gap_list = []
        for i in range(len(eig_vals) - 1):
            gap = float(eig_vals[i + 1]) - float(eig_vals[i])
            # eig_vals_gap_list.append(float(eig_vals[i + 1]) - float(eig_vals[i]))
            eig_vals_gap_list.append(gap)

        print(eig_vals_gap_list, file=sys.stderr)
        return eig_vals_gap_list
