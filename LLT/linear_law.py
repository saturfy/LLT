"""This sub package collects the functions and classes which calculate the linear laws and generate the LLT features. """
import numpy as np


def embedding(sample, num_lags, sampling_step=1, base_step=1, lag_step=1):
    """
    Takes time series and creates a time delay embedding matrix from it.

    Parameters
    ----------
    sample : numpy.array shape=(lenght of samples,)
        array containing a time series
    num_lags : int
        The number of lagged datapoints corresponding to every base point. num_lags +1 will
        be the size of samples in the dataset created from the time series (lagges points +
        base point) which is also called the depth of the embedding.
    sampling_step : int
        The time series will be resampled with this step size. If it is 2 or 3 every second or
        third point will be taken from the original.
    base_step : int
        The step size which determines the position of the base points. It starts from the last
        point of the time series and takes every 'base_step'th point as a base point.
    lag_step : int
        The step size which determines the position of the lag points. It starts from the base
        points and moves backward in time.
    Returns
    -------
    nump.ndarray
        Time embedding matrix, shape(number of base points, number of lags+1) The time series is converted
        into this matrix where Y[n,k] = y(t = n * delta t - k * delta t). Every row corresponds to
        a slice of the resampled time series which starts at a given base point and contains
        lagged datapoints moving backward in time.

    Notes
    -----
    All of the prodecures are done using the resampled time series. Every selection starts from
    the last point (most recent) of the time series and moves backward.

    STEP is moving one unit in the resampled series, so lag_step=2 means taking every second point
    as lag point. (these are sampling_step * lag_step  steps in the original series)

    """
    # safe copy
    y = np.copy(sample)

    # reverse and resample the time series
    y_resampled = y[-1::-sampling_step]

    # base points
    base_p_indexes = np.arange(0, len(y_resampled), 1)[0: -num_lags * lag_step:base_step]
    base_p_num = len(base_p_indexes)

    # --- create embedding matrix ---
    Y = np.zeros((base_p_num, num_lags + 1))

    for i, bp in enumerate(base_p_indexes):
        Y[i, :] = y_resampled[bp:bp + (num_lags + 1) * lag_step:lag_step]

    return Y


def embedding_dataset(samples, num_lags, concatenate=True, sampling_step=1, base_step=1, lag_step=1):
    """
    Takes a set of time series and creates one large time delay embedding matrix from them.
    if concatenate is Ture:
    The time embedding matrices of the individual samples are concatenated into a large block
    matrix where the matrixes corresponding to individual samples are augmented by new rows
    containing the next sample. This way the large block matrix is very high and has
    as many columns as many time delay steps were taken. Each columns corresponds to
    the same time delay.
    if concatenate is False:
    The time embedding matrices of individual samples is not conatenated together. The original
    strucuter of the dataset is kept, except the time series samples are switched to their time
    embedding matrices.

    Parameters
    ----------
    samples : numpy.array shape=(number of samples, lenght of samples)
        array containing a set of time series
    num_lags : int
        The number of lagged datapoints corresponding to every base point. num_lags +1 will
        be the size of samples in the dataset created from the time series (lagges points +
        base point) which is also called the lenght of the embedding.
    sampling_step : int
        The time series will be resampled with this step size. If it is 2 or 3 every second or
        third point will be taken from the original.
    base_step : int
        The step size which determines the position of the base points. It starts from the last
        point of the time series and takes every 'base_step'th point as a base point.
    lag_step : int
        The step size which determines the position of the lag points. It starts from the base
        points and moves backward in time.
    Returns
    -------
    numpy.ndarray
    if concatenate is True: shape=(-1,time embedding depth)
    if concatenate is False: shape=(number of samples, -1, time embedding length)

    """
    # time delayed dataset
    Y = []
    for sample in samples:
        y = embedding(sample, num_lags=num_lags, sampling_step=sampling_step, base_step=base_step, lag_step=lag_step)
        if concatenate:
            Y.extend(y)
        else:
            Y.append(y)

    return np.array(Y)


class linear_model:
    """
    Fits a model of linear law and calculates properties.

    Attributes
    ----------
    learning_set : numpy.ndarray
            shape(number of samples, embedding length)
            The embedding matrix of a dataset. Columns corresponding to time delays and rows
            contain the different time delay slices.
    PCA_eigenvalues : numpy.ndarray
        array of the eigenvalues coming from the PCA decomposition in increasing order.
    PCA_eigenvectors : numpy.ndarray
        array of the eigenvectors corresponding to the PCA decomposition. Vectors are ordered according
        to the value their corresponding eigenvalues.
    linear_law : numpy.ndarray
        Array of the linear law corresponding to the smallest eigenvalue.
    score : float
        Measures how well the linera law fits the data. Lower the better.
        In a case of a perfect fit the linera law maps every line of the
        learning set to zero. In practice mapping is just close to zero.
        Score is: applying the linear law to the learning set and
        taking the square root of the variance of the resulting vector.
    _num_lags : int
        The number of lagged points in embedding generated from the data_set
    _sampling_step : int
        The resampling step applied on the data set
    _base_step : int
        The step size which determines the position of the base points while
        generating embedding from the data set. It starts from the last
        point of the time series and takes every 'base_step'th point as a base point.
    _lag_step : int
        The step size which determines the position of the lag points while
        generating embedding from the datas set. It starts from the base points
        and moves backward in time.
    """

    def __init__(self, data_set, num_lags, sampling_step=1, base_step=1, lag_step=1):
        """
        Creates a class instance and initializes with the learning set.

        Parameters
        ----------
        data_set : numpy.ndarray
            shape(number of samples, signal length)
            The time series data of the peaks which will be used to determine the linear law.
            This can be considered the learning set.
        num_lags : int
            The number of lagged datapoints corresponding to every base point. num_lags +1 will
            be the size of samples in the dataset created from the time series (lagges points +
            base point) which is also called the lenght of the embedding.
        sampling_step : int
            The time series will be resampled with this step size. If it is 2 or 3 every second or
            third point will be taken from the original.
        base_step : int
            The step size which determines the position of the base points. It starts from the last
            point of the time series and takes every 'base_step'th point as a base point.
        lag_step : int
            The step size which determines the position of the lag points. It starts from the base
            points and moves backward in time.

        """
        # create a time delay embedding from the dataset
        self.learning_set = embedding_dataset(data_set, num_lags=num_lags,
                                              sampling_step=sampling_step,
                                              base_step=base_step,
                                              lag_step=lag_step)

        # create the attributes
        self.PCA_eigenvalues = None
        self.PCA_eigenvectors = None
        self.linear_law = None
        self.score = None
        self._num_lags = num_lags
        self._sampling_step = sampling_step
        self._base_step = base_step
        self._lag_step = lag_step

    def fit(self):
        """
        Calculates the linear model from learning set.

        Parameters
        ----------
        None

        Returns
        -------
        None

        """

        # --- Create PCA matrix ---
        PCA_matrix = self.learning_set.T @ self.learning_set / len(self.learning_set[:, 0])

        # --- linear law ---
        # calculating the eigensystem
        eigenvals, eigenvectors = np.linalg.eig(PCA_matrix)
        idx_list = np.argsort(eigenvals)
        self.PCA_eigenvalues = eigenvals[idx_list]
        self.PCA_eigenvectors = [eigenvectors[:, i] for i in idx_list]

        # --- search for imaginary eigenvalues ---
        imag_list = np.where(np.imag(eigenvals) > 0)[0]
        if len(imag_list) != 0:
            raise ValueError('Numerical precision error, there are complex eigenvalues.')

        # --- minimal eigenvalue-vector pair ---
        minimal_index = np.argmin(eigenvals)
        w = eigenvectors[:, minimal_index]
        self.linear_law = w

        # --- calculate the average accuracy of the linear law ---
        self.score = np.sqrt((self.learning_set @ self.linear_law).var())

    def feature_transform(self, data_set, normalize=0.95):
        """
        Generates features from the data set of using the linear law.
        The original structure of the  dataset will be preserved
        (sample indexes remain valid), but instead of the individual
        samples, the output will contain thecorresponding features.

        Parameters
        ----------
        data_set : numpy.ndarray
            shape(number of samples, signal length)
            The time series data of the peaks which will be used to determine the linear law.
            This can be considered the learning set.
        normalize : float
            Determines the normalization constant for the features.
            if None: no normalization happens
            if float: percentile
            the normalization constant is determined as the given percentile in the abs(dataset).
        """
        transformed_data = []
        # --- data set embedding ---
        embeddings = embedding_dataset(data_set,
                                       concatenate=False,
                                       num_lags=self._num_lags,
                                       sampling_step=self._sampling_step,
                                       base_step=self._base_step,
                                       lag_step=self._lag_step)

        # --- normlization constant ---
        if normalize is not None:
            N = np.percentile(np.abs((self.learning_set @ self.linear_law)), normalize*100)

        for embedding in embeddings:
            features = embedding @ self.linear_law

            # --- normalize ---
            if normalize is not None:
                features = features / N

            transformed_data.append(features)

        return np.array(transformed_data)
