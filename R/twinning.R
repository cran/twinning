data_format = function(data)
{
	if(anyNA(data))
	{
		stop("dataset contains missing value(s)")
	}

	D = matrix(, nrow=nrow(data))
	for(j in 1:ncol(data))
	{
		if(is.factor(data[, j]))
		{
			factor = unique(data[, j])
			if(length(factor) == 1)
			{
				next
			}

			factor_helm = contr.helmert(length(factor))
			helm = factor_helm[match(data[, j], factor), ]
			D = cbind(D, helm)
		}
		else
		{
			if(is.numeric(data[, j]))
			{
				if(max(data[, j]) == min(data[, j]))
				{
					next
				}

				D = cbind(D, data[, j])
			}
			else
			{
				stop("dataset constains non-numeric non-factor column(s)")
			}
		}
	}

	D = D[, -1]
	D = scale(D)
	return(D)
}


#' Partition datasets into statistcally similar twin sets
#' 
#' \code{twin()} implements the twinning algorithm presented in Vakayil and Joseph (2022). A partition of the dataset is returned, such that the resulting two disjoint sets, termed as \emph{twins}, are distributed similar to each other, as well as the whole dataset. Such a partition is an optimal training-testing split (Joseph and Vakayil, 2021) for training and testing statistical and machine learning models, and is model-independent. The statistical similarity also allows one to treat either of the twins as a compression (lossy) of the dataset for tractable model building on Big Data. 
#'
#' @param data The dataset including both the predictors and response(s); should not contain missing values, and only numeric and/or factor column(s) are allowed.  
#' @param r An integer representing the inverse of the splitting ratio, e.g., for an 80-20 partition, \code{r = 1 / 0.2 = 5}.
#' @param u1 Index of the data point from where twinning starts; if not provided, twinning starts from a random point in the dataset. Fixing \code{u1} makes twinning deterministic, i.e., the same twins are returned.
#' @param format_data If set to \code{TRUE}, constant columns in \code{data} are removed, factor columns are converted to numerical using Helmert coding, and then the columns are scaled to zero mean and unit standard deviation. If set to \code{FALSE}, the user is expected to perform data pre-processing.
#' @param leaf_size Maximum number of elements in the leaf-nodes of the \emph{kd}-tree.
#'
#' @return Indices of the smaller twin.
#'
#' @details The twinning algorithm requires nearest neighbor queries that are performed using a \emph{kd}-tree. The \emph{kd}-tree implementation in the \code{nanoflann} (Blanco and Rai, 2014) C++ library is used.
#'
#' @export
#' @examples
#' ## 1. An 80-20 partition of a numeric dataset
#' X = rnorm(n=100, mean=0, sd=1)
#' Y = rnorm(n=100, mean=X^2, sd=1)
#' data = cbind(X, Y)
#' twin1_indices = twin(data, r=5) 
#' twin1 = data[twin1_indices, ]
#' twin2 = data[-twin1_indices, ]
#' plot(data, main="Smaller Twin")
#' points(twin1, col="green", cex=2)
#'
#' ## 2. An 80-20 split of the iris dataset
#' twin1_indices = twin(iris, r=5)
#' twin1 = iris[twin1_indices, ]
#' twin2 = iris[-twin1_indices, ]
#'
#' @references
#' Vakayil, A., & Joseph, V. R. (2022). Data Twinning. Statistical Analysis and Data Mining: The ASA Data Science Journal, to appear. arXiv preprint arXiv:2110.02927.
#'
#' Joseph, V. R., & Vakayil, A. (2021). SPlit: An Optimal Method for Data Splitting. Technometrics, 1-11. doi:10.1080/00401706.2021.1921037.
#'
#' Blanco, J. L. & Rai, P. K. (2014). nanoflann: a C++ header-only fork of FLANN, a library for nearest neighbor (NN) with kd-trees. https://github.com/jlblancoc/nanoflann.


twin = function(data, r, u1=NULL, format_data=TRUE, leaf_size=8)
{
	if(r %% 1 != 0 | r < 2 | r > nrow(data) / 2)
	{
		stop("r should be an integer such that 2 <= r <= nrow(data)/2")
	}

	if(format_data)
	{
		D = data_format(data)
	}
	else if(is.numeric(data) & is.matrix(data))
	{
		D = data
	}
	else
	{
		stop("data should be a numeric matrix")
	}

	return(twin_cpp(D, r, ifelse(is.null(u1), sample(nrow(D), 1), u1), leaf_size))
}


#' Partition datasets into multiple statistcally similar disjoint sets
#' 
#' \code{multiplet()} extends \code{\link{twin}()} to partition datasets into multiple statistically similar disjoint sets, termed as \emph{multiplets}, under the three different strategies described in Vakayil and Joseph (2022).
#'
#' @param data The dataset including both the predictors and response(s); should not contain missing values, and only numeric and/or factor column(s) are allowed.  
#' @param k The desired number of multiplets.
#' @param strategy An integer either 1, 2, or 3 referring to the three strategies for generating multiplets. Strategy 2 perfroms best, but requires \code{k} to be a power of 2. Strategy 3 is computatioanlly inexpensive, but performs worse than strategies 1 and 2.
#' @param format_data If set to \code{TRUE}, constant columns in \code{data} are removed, factor columns are converted to numerical using Helmert coding, and then the columns are scaled to zero mean and unit standard deviation. If set to \code{FALSE}, the user is expected to perform data pre-processing.
#' @param leaf_size Maximum number of elements in the leaf-nodes of the \emph{kd}-tree.
#'
#' @return List with the multiplet id, ranging from 1 to \code{k}, for each row in \code{data}.
#'
#' @export
#' @examples
#' ## 1. Generating 10 multiplets of a numeric dataset
#' X = rnorm(n=100, mean=0, sd=1)
#' Y = rnorm(n=100, mean=X^2, sd=1)
#' data = cbind(X, Y)
#' multiplet_idx = multiplet(data, k=10) 
#' multiplet_1 = data[which(multiplet_idx == 1), ]
#' multiplet_10 = data[which(multiplet_idx == 10), ]
#'
#' ## 2. Generating 4 multiplets of the iris dataset using strategy 2
#' multiplet_idx = multiplet(iris, k=4, strategy=2)
#' multiplet_1 = iris[which(multiplet_idx == 1), ]
#' multiplet_4 = iris[which(multiplet_idx == 4), ]
#'
#' @references
#' Vakayil, A., & Joseph, V. R. (2022). Data Twinning. Statistical Analysis and Data Mining: The ASA Data Science Journal, to appear. arXiv preprint arXiv:2110.02927.
#'
#' Blanco, J. L. & Rai, P. K. (2014). nanoflann: a C++ header-only fork of FLANN, a library for nearest neighbor (NN) with kd-trees. https://github.com/jlblancoc/nanoflann.


multiplet = function(data, k, strategy=1, format_data=TRUE, leaf_size=8)
{
	if(k %% 1 != 0 | k < 2 | k > nrow(data) / 2)
	{
		stop("k should be an integer such that 2 <= k <= nrow(data)/2")
	}

	if(!strategy %in% c(1, 2, 3))
	{
		stop("strategy should be 1, 2, or 3")
	}

	if(format_data)
	{
		D = data_format(data)
	}
	else if(is.numeric(data) & is.matrix(data))
	{
		D = data
	}
	else
	{
		stop("data should be a numeric matrix")
	}
	
	N = nrow(D)

	if(strategy == 1)
	{
		D = cbind(1:nrow(D), D)
		i = 0
		folds = matrix(, ncol=2)
		while(TRUE)
		{	
			multiplet_i = twin_cpp(D[, 2:ncol(D)], k - i, sample(nrow(D), 1), leaf_size)
			folds = rbind(folds, cbind(D[multiplet_i, 1], rep(i + 1, length(multiplet_i))))
			
			D = D[-multiplet_i, ]
			if(nrow(D) <= N / k)
			{
				folds = rbind(folds, cbind(D[, 1], rep(i + 2, nrow(D))))
				break
			}

			i = i + 1
		}

		folds = folds[-1, ]
		return(folds[order(folds[, 1]), 2])
	}

	if(strategy == 2)
	{
		isPowerOf2 = function(x) 
		{
			n1s = sum(as.numeric(intToBits(x)))
			if(n1s == 1) 
			{
				return(TRUE)
			} 
			else 
			{
				return(FALSE)
			}
		}

		if(!isPowerOf2(k))
		{
			stop("strategy 2 requires k to be a power of 2")
		}

		D = cbind(1:nrow(D), D)
		folds = matrix(, ncol=2)
		i = 0
		equal_twin = function(D_)
		{
			if(nrow(D_) <= ceiling(N / k))
			{
				folds <<- rbind(folds, cbind(D_[, 1], rep(i + 1, nrow(D_))))
				i <<- i + 1
			}
			else
			{
				equal_twin_i = twin_cpp(D_[, 2:ncol(D_)], 2, sample(nrow(D_), 1), leaf_size)
				equal_twin(D_[equal_twin_i, ])
				equal_twin(D_[-equal_twin_i, ])
			}
		}

		equal_twin(D)
		folds = folds[-1, ]
		return(folds[order(folds[, 1]), 2])
	}

	if(strategy == 3)
	{
		sequence = multipletS3_cpp(D, 2, sample(nrow(D), 1), leaf_size)
		folds = cbind(sequence, rep(1:k, ceiling(N / k))[1:length(sequence)])
		return(folds[order(folds[, 1]), 2])
	}
}


#' Energy distance computation
#' 
#' \code{energy()} computes the energy distance (Székely and Rizzo, 2013)  between a given dataset and a set of points in same dimensions. 
#'
#' @param data The dataset including both the predictors and response(s). A numeric matrix is expected. If the dataset has factor columns, the user is expected to convert them to numeric using a coding method.
#' @param points The set of points for which the energy distance with respect to \code{data} is to be computed. A numeric matrix is expected.
#'
#' @return Energy distance.
#'
#' @details Smaller the energy distance, the more statistically similar the set of points is to the given dataset. The minimizer of energy distance is known as support points (Mak and Joseph, 2018), which is the basis of the twinning method. Computing energy distance between \code{data} and \code{points} involves Euclidean distance calculations among the rows of \code{data}, among the rows of \code{points}, and between the rows of \code{data} and \code{points}. Since, \code{data} serves as the reference, the distance calculations among the rows of \code{data} are ignored for efficiency. Before computing the energy distance, the columns of \code{data} are scaled to zero mean and unit standard deviation. The mean and standard deviation of the columns of \code{data} are used to scale the respective columns in \code{points}.
#'
#' @export
#' @examples
#' ## Energy distance between a dataset and a random sample
#' X = rnorm(n=100, mean=0, sd=1)
#' Y = rnorm(n=100, mean=X^2, sd=1)
#' data = cbind(X, Y)
#' energy(data, data[sample(100, 20), ])
#'
#' @references
#' Vakayil, A., & Joseph, V. R. (2022). Data Twinning. Statistical Analysis and Data Mining: The ASA Data Science Journal, to appear. arXiv preprint arXiv:2110.02927.
#' 
#' Székely, G. J., & Rizzo, M. L. (2013). Energy statistics: A class of statistics based on distances. Journal of statistical planning and inference, 143(8), 1249-1272.
#'
#' Mak, S. & Joseph, V. R. (2018). Support Points. Annals of Statistics, 46, 2562-2592.


energy = function(data, points)
{
	if(is.numeric(data) & is.matrix(data) & is.numeric(points) & is.matrix(points))
	{
		if(ncol(data) != ncol(points))
		{
			stop("data and points should be in same dimensions")
		}

		D = scale(data)
		sp = scale(points, center=attributes(D)$"scaled:center", scale=attributes(D)$"scaled:scale")
		return(energy_cpp(D, sp))
	}
	else
	{
		stop("both data and points should be numeric matrices")
	}
}












