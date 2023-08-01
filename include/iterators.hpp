#include <vector>
#include <tuple>
#include <iterator>

template<typename I>
class boxed_iterator {
    I i;

public:
    typedef I difference_type;
    typedef I value_type;
    typedef I pointer;
    typedef I reference;
    typedef std::random_access_iterator_tag iterator_category;

    boxed_iterator(I i) : i{i} {}

    bool operator==(boxed_iterator<I> &other) { return i == other.i; }
    I operator-(boxed_iterator<I> &other) { return i - other.i; }
    I operator++() { return i++; }
    I operator*() { return i; }
};