#ifndef align_h
#define align_h
#include <cstdlib>


template <typename T, size_t NALIGN>
struct AlignedAllocator : public std::allocator<T> {
  //template <typename U>
  //struct rebind {
  //  typedef AlignedAllocator<U, NALIGN> other;
  //};

  T * allocate(size_t n) {
    void *ptr = nullptr;
    ptr = alligned_alloc(NALIGN, n * sizeof(T));
    return reinterpret_cast<T*>(ptr);
  }

  void deallocate(T* p, size_t) {
    return std::free(p);
  }
};
#endif
