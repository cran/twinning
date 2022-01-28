// [[Rcpp::plugins("cpp11")]]
#include <memory>
#include <vector>
#include <Rcpp.h>
#include <cmath>
#include "nanoflann.hpp"


class DF
{
private:
	std::shared_ptr<Rcpp::NumericMatrix> df_;

public:
	void import_data(Rcpp::NumericMatrix& df)
	{
		df_ = std::make_shared<Rcpp::NumericMatrix>(Rcpp::transpose(df));
	}

	std::size_t kdtree_get_point_count() const
	{
		return df_->cols();
	}

	double kdtree_get_pt(const std::size_t idx, const std::size_t dim) const 
	{
		return (*df_)(dim, idx);
	}

	const double* get_row(const std::size_t idx) const
	{
		return &(*df_)(0, idx);
	}

	template <class BBOX>
	bool kdtree_get_bbox(BBOX&) const 
	{ 
		return false; 
	}
};


typedef nanoflann::KDTreeSingleIndexDynamicAdaptor<nanoflann::L2_Adaptor<double, DF>, DF, -1, std::size_t> kdTree;


class KDTree
{
private:
	const std::size_t dim_;
	const std::size_t N_;
	const std::size_t r_;
	const std::size_t u1_;
	const std::size_t leaf_size_;
	DF data_;

public:
	KDTree(Rcpp::NumericMatrix& data, std::size_t r, std::size_t u1, std::size_t leaf_size) : 
	dim_(data.cols()), N_(data.rows()), r_(r), u1_(u1 - 1), leaf_size_(leaf_size)
	{
		data_.import_data(data);
	}

	std::vector<std::size_t> twin()
	{
		kdTree tree(dim_, data_, nanoflann::KDTreeSingleIndexAdaptorParams(leaf_size_));
		nanoflann::KNNResultSet<double> resultSet(r_);
		std::size_t *index = new std::size_t[r_];
		double *distance = new double[r_];

		nanoflann::KNNResultSet<double> resultSet_next_u(1);
		std::size_t index_next_u;
		double distance_next_u;

		std::vector<std::size_t> indices;
		indices.reserve(N_ / r_ + 1);
		std::size_t position = u1_;

		while(true)
		{
			resultSet.init(index, distance);
			tree.findNeighbors(resultSet, data_.get_row(position), nanoflann::SearchParams());
			indices.push_back(index[0] + 1);

			for(std::size_t i = 0; i < r_; i++)
				tree.removePoint(index[i]);

			resultSet_next_u.init(&index_next_u, &distance_next_u);
			tree.findNeighbors(resultSet_next_u, data_.get_row(index[r_ - 1]), nanoflann::SearchParams());	
			position = index_next_u;

			if(N_ - indices.size() * r_ <= r_)
			{
				indices.push_back(position + 1);
				break;
			}
		}

		delete[] index;
		delete[] distance;

		return indices;
	}

	std::vector<std::size_t> multipletS3()
	{
		kdTree tree(dim_, data_, nanoflann::KDTreeSingleIndexAdaptorParams(leaf_size_));
		nanoflann::KNNResultSet<double> resultSet(r_);
		std::size_t* index = new std::size_t[r_];
		double* distance = new double[r_];

		nanoflann::KNNResultSet<double> resultSet_next_u(1);
		std::size_t index_next_u;
		double distance_next_u;

		std::vector<std::size_t> sequence;
		sequence.reserve(N_);
		std::size_t position = u1_;
		
		while(sequence.size() != N_)
		{
			if(sequence.size() > N_ - r_)
			{
				std::size_t r_f = N_ - sequence.size();
				nanoflann::KNNResultSet<double> resultSet_f(r_f);
				std::size_t* index_f = new std::size_t[r_f];
				double* distance_f = new double[r_f]; 

				resultSet_f.init(index_f, distance_f);
				tree.findNeighbors(resultSet_f, data_.get_row(position), nanoflann::SearchParams());

				for(std::size_t i = 0; i < r_f; i++)
					sequence.push_back(index_f[i] + 1);

				delete[] index_f;
				delete[] distance_f;

				break;
			}

			resultSet.init(index, distance);
			tree.findNeighbors(resultSet, data_.get_row(position), nanoflann::SearchParams());

			for(std::size_t i = 0; i < r_; i++)
			{
				sequence.push_back(index[i] + 1);
				tree.removePoint(index[i]);
			}

			resultSet_next_u.init(&index_next_u, &distance_next_u);
			tree.findNeighbors(resultSet_next_u, data_.get_row(index[r_ - 1]), nanoflann::SearchParams());	
			position = index_next_u;	
		}

		delete[] index;
		delete[] distance;

		return sequence;
	}
};


// [[Rcpp::export]]
std::vector<std::size_t> twin_cpp(Rcpp::NumericMatrix& data, std::size_t r, std::size_t u1, std::size_t leaf_size=8)
{
	KDTree tree(data, r, u1, leaf_size);
	return tree.twin();
}

// [[Rcpp::export]]
std::vector<std::size_t> multipletS3_cpp(Rcpp::NumericMatrix& data, std::size_t r, std::size_t u1, std::size_t leaf_size=8)
{
	KDTree tree(data, r, u1, leaf_size);
	return tree.multipletS3();
}

// [[Rcpp::export]]
double energy_cpp(Rcpp::NumericMatrix& data, Rcpp::NumericMatrix& points)
{
	DF D, sp;
	std::size_t dim = data.cols();
	std::size_t N = data.rows();
	std::size_t n = points.rows();

	D.import_data(data);
	sp.import_data(points);
		
	std::vector<double> ed_1;
	std::vector<double> ed_2;
	ed_1.resize(n);
	ed_2.resize(n);

	#pragma omp parallel for
	for(std::size_t i = 0; i < n; i++)
	{
		const double* u_i = sp.get_row(i);

		double distance_sum = 0.0;
		double inner_sum = 0.0;
		for(std::size_t j = 0; j < N; j++)
		{
			const double* z_j = D.get_row(j);

			inner_sum = 0.0;
			for(std::size_t k = 0; k < dim; k++)
				inner_sum += std::pow(*(u_i + k) - *(z_j + k), 2);

			distance_sum += std::sqrt(inner_sum);
		}

		ed_1[i] = distance_sum;

		distance_sum = 0.0;
		for(std::size_t j = 0; j < n; j++)
			if(j != i)
			{
				const double* u_j = sp.get_row(j);

				inner_sum = 0.0;
				for(std::size_t k = 0; k < dim; k++)
					inner_sum += std::pow(*(u_i + k) - *(u_j + k), 2); 

				distance_sum += std::sqrt(inner_sum);
			}

		ed_2[i] = distance_sum;
	}

	double sum1 = 0.0;
	double sum2 = 0.0;
	for(std::size_t i = 0; i < n; i++)
	{
		sum1 += ed_1[i];
		sum2 += ed_2[i];
	}

	return 2.0 * sum1 / (N * n) - sum2 / (n * n);
}
