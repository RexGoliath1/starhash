#include <Eigen/Dense>

// Define dynamic eigen binary arrays
typedef Eigen::Array<bool,Eigen::Dynamic,1> ArrayXb;

class logical
{
private:
    const Eigen::Index new_size;
    Eigen::Array<Eigen::Index, Eigen::Dynamic, 1> old_inds;

public:
    logical(const Eigen::Array<bool, Eigen::Dynamic, 1> &keep) : new_size(keep.count()), old_inds(new_size)
    {
        for (Eigen::Index i = 0, j = 0; i < keep.size(); i++)
            if (keep(i))
                old_inds(j++) = i;
    }
    Eigen::Index size() const { return new_size; }
    Eigen::Index operator[](Eigen::Index new_ind) const { return old_inds(new_ind); }
};