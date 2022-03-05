#pragma once
/******************************************************************************
 *
 * ExaFMM
 * A high-performance fast multipole method library with C++ and
 * python interfaces.
 *
 * Originally Tingyu Wang, Rio Yokota and Lorena A. Barba
 * Modified by HJA Bird
 *
 ******************************************************************************/
#ifndef INCLUDE_EXAFMM_RELATIVE_COORDS_H_
#define INCLUDE_EXAFMM_RELATIVE_COORDS_H_

#include <Eigen/Dense>

#include <array>

namespace ExaFMM {
	typedef Eigen::Vector3i ivec3;  //!< Vector of 3 int types
namespace detail {

// We need a constexpr version of abs.
constexpr int abs(int x) { return x > 0 ? x : -x; }

// Problems with std::array mean we need our own version. (MSVC debug builds
// include non-constexpr debugging features).
template<typename ElemT, int Size>
class array {
private:
	// A raw array to hold the data.
	ElemT m_data[Size];

public:
	constexpr ElemT& operator[](size_t i) noexcept { return m_data[i]; }
	constexpr const ElemT& operator[](size_t i) const noexcept { return m_data[i]; }
	constexpr size_t size() const noexcept { return Size; }
	constexpr ElemT* data() noexcept { return m_data; }
};

/** The mapping definition defines a set points within a cube. The points are
* placed on 3D grid-points from [-includeBoxRadius:pointStride:includeBoxRadius]
* in each dimension, except where they lie within (not <=, <) excludeBoxRadius.
**/
template<class CoordType>
class relative_coords_mapping_def;

/** FMM interaction types.
**/
class coords_M2M;
class coords_L2L;
class coords_M2L;
class coords_M2L_helper;

/** Information regarding the mapping of the coordinate indexes to grid 
 * locations for FMM interaction types.
* @tparam CoordType The FMM interaction type.
**/
template<class CoordType>
class relative_coord_mapping {
protected:
	// The mapping definition associated with the coord type.
	using mapping_def_t = relative_coords_mapping_def<CoordType>;

	// Number of points in idx->ivec3 map.
	static constexpr int numPointsForward = 
		mapping_num_points_forward<mapping_def_t>();
	// Number of points in ivec3->idx map
	static constexpr int numPointsBackward = 
		mapping_num_points_backward<mapping_def_t>();

	// Mappings. C-style arrays are used in favour of std::array due to 
	// constexpr issues with debug builds in MSVC.
	// Map from idx->ivec3
	static constexpr array<int[3], numPointsForward> s_relCoords =
		create_coords_array_forward<mapping_def_t>();
	// Map from ivec3->idx.
	static constexpr array<size_t, numPointsBackward> s_hashCoords =
		create_coords_array_backward<mapping_def_t>();

public:
	// Obtain an ivec3 associated with a index
	constexpr ivec3 operator[](size_t mapIdx) { 
		return ivec3{ s_relCoords[mapIdx] }; 
	}

	// Obtain an index from an ivec3. Bad inputs return 123456789.
	static constexpr size_t hash(ivec3 coord) {
		size_t idx = coord_to_rel_coord_map_ijk_to_idx<mapping_def_t>(
			coord[0], coord[1], coord[2]);
		return s_hashCoords[idx];
	}

	// Obtain the number of points in the forward mapping.
	static constexpr size_t size() { return numPointsForward; }
};

/// Query whether a grid point {i,j,k} is included within a relative coord 
/// mapping.
template<class RelCoordsMappingDef>
constexpr bool point_in_forward_map(int i, int j, int k) {
	using map_t = RelCoordsMappingDef;
	int incRad = map_t::includeBoxRadius;
	int excRad = map_t::excludeBoxRadius;
	bool inInc = (abs(i) <= incRad) && (abs(j) <= incRad) && (abs(k) <= incRad);
	bool inExc = (abs(i) < excRad) && (abs(j) < excRad) && (abs(k) < excRad);
	bool onStrideI = ((i + incRad) % map_t::pointStride) == 0;
	bool onStrideJ = ((j + incRad) % map_t::pointStride) == 0;
	bool onStrideK = ((k + incRad) % map_t::pointStride) == 0;
	return (onStrideI && onStrideJ && onStrideK) && inInc && !inExc;
}

/// Create the forward mapping definition array. 
template<class RelCoordsMappingDef>
constexpr auto create_coords_array_forward() {
	using map_t = RelCoordsMappingDef;
	int incR = map_t::includeBoxRadius;
	int excR = map_t::excludeBoxRadius;
	int stride = map_t::pointStride;
	const int numPoints = mapping_num_points_forward<map_t>();
	array<int[3], numPoints> retVal = {};
	int idx{0};
	for (int k = -incR; k <= incR; k += stride) {
		for (int j = -incR; j <= incR; j += stride) {
			for (int i = -incR; i <= incR; i += stride) {
				if (point_in_forward_map<map_t>(i, j, k)) {
					retVal[idx][0] = i;
					retVal[idx][1] = j;
					retVal[idx][2] = k;
					idx += 1;
				}
			}
		}
	}
	return retVal;
}

/// How many points are in the forward mapping - the size of the array.
template<class RelCoordsMappingDef>
constexpr int mapping_num_points_forward() {
	using map_t = RelCoordsMappingDef;
	int cubeSideInc = map_t::includeBoxRadius * 2 / map_t::pointStride + 1;
	int cubeSideExc = map_t::excludeBoxRadius * 2 / map_t::pointStride - 1;
	int numPointsInc = cubeSideInc * cubeSideInc * cubeSideInc;
	int numPointsExc = cubeSideExc * cubeSideExc * cubeSideExc;
	return numPointsInc - numPointsExc;
}

/// Create the backwards mapping from {i, j, k} indexes to a linear index 
/// corresponding to that position in the forward map.
template<class RelCoordsMappingDef>
constexpr auto create_coords_array_backward() {
	using map_t = RelCoordsMappingDef;
	const int numPoints = mapping_num_points_backward<map_t>();
	array<size_t, numPoints> retVal = {};
	for (int i{ 0 }; i < retVal.size(); ++i) {
		retVal[i] = 123456789;
	}
	int incR = map_t::includeBoxRadius;
	int stride = map_t::pointStride;
	int idx{ 0 };
	for (int k = -incR; k <= incR; k += stride) {
		for (int j = -incR; j <= incR; j += stride) {
			for (int i = -incR; i <= incR; i += stride) {
				if (point_in_forward_map<map_t>(i, j, k)) {
					size_t mapIdx = coord_to_rel_coord_map_ijk_to_idx<map_t>(
						i, j, k);
					retVal[mapIdx] = idx;
					idx += 1;
				}
			}
		}
	}
	return retVal;
}

/// Obtain a linear index in the backwards map from {i, j, k}
template<class RelCoordsMappingDef>
constexpr size_t coord_to_rel_coord_map_ijk_to_idx(int i, int j, int k) {
	using map_t = RelCoordsMappingDef;
	int incR = map_t::includeBoxRadius;
	int stride = map_t::pointStride;
	int cubeSideInc = incR * 2 / stride + 1;
	int ak{ (k + incR) / stride }, 
		aj{ (j + incR) / stride }, 
		ai{ (i + incR) / stride };
	return ak + cubeSideInc * (aj + cubeSideInc * ai);
}

/// Size of the backwards map. Map is sparse to make mapping from {i, j, k} easier.
template<class RelCoordsMappingDef>
constexpr int mapping_num_points_backward() {
	using map_t = RelCoordsMappingDef;
	const int cubeSideInc = map_t::includeBoxRadius * 2 / map_t::pointStride + 1;
	const int numPointsInc = cubeSideInc * cubeSideInc * cubeSideInc;
	return numPointsInc;
}

template<>
class relative_coords_mapping_def<coords_M2M> {
public:
	static constexpr int includeBoxRadius{ 1 };
	static constexpr int excludeBoxRadius{ 1 };
	static constexpr int pointStride{ 2 };
};

template<>
class relative_coords_mapping_def<coords_L2L> {
public:
	static constexpr int includeBoxRadius{ 1 };
	static constexpr int excludeBoxRadius{ 1 };
	static constexpr int pointStride{ 2 };
};

template<>
class relative_coords_mapping_def<coords_M2L_helper> {
public:
	static constexpr int includeBoxRadius{ 3 };
	static constexpr int excludeBoxRadius{ 2 };
	static constexpr int pointStride{ 1 };
};

template<>
class relative_coords_mapping_def<coords_M2L> {
public:
	static constexpr int includeBoxRadius{ 1 };
	static constexpr int excludeBoxRadius{ 1 };
	static constexpr int pointStride{ 1 };
};

} // namespace detail
} // namespace ExaFMM


#endif //INCLUDE_EXAFMM_RELATIVE_COORDS_H_
