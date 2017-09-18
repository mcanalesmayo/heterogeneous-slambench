/*

 Copyright (c) 2011-2013 Gerhard Reitmayr, TU Graz

 Copyright (c) 2014 University of Edinburgh, Imperial College, University of Manchester.
 Developed in the PAMELA project, EPSRC Programme Grant EP/K008730/1

 This code is licensed under the MIT License.

 */

/************** TYPES ***************/

typedef struct sVolume {
	uint3 size;
	float3 dim;
	__global short2 * data;
} Volume;

typedef struct sMatrix4 {
	float4 data[4];
} Matrix4;

/************** FUNCTIONS ***************/

inline float vs(const uint3 pos, const Volume v) {
	return v.data[pos.x + pos.y * v.size.x + pos.z * v.size.x * v.size.y].x;
}

inline float interp(const float3 pos, const Volume v) {
	const float3 scaled_pos = (float3)((pos.x * v.size.x / v.dim.x) - 0.5f,
			(pos.y * v.size.y / v.dim.y) - 0.5f,
			(pos.z * v.size.z / v.dim.z) - 0.5f);
	float3 basef = (float3)(0);
	const int3 base = convert_int3(floor(scaled_pos));
	const float3 factor = (float3)(fract(scaled_pos, (float3 *) &basef));
	const int3 lower = max(base, (int3)(0));
	const int3 upper = min(base + (int3)(1), convert_int3(v.size) - (int3)(1));
	return (((vs((uint3)(lower.x, lower.y, lower.z), v) * (1 - factor.x)
			+ vs((uint3)(upper.x, lower.y, lower.z), v) * factor.x)
			* (1 - factor.y)
			+ (vs((uint3)(lower.x, upper.y, lower.z), v) * (1 - factor.x)
					+ vs((uint3)(upper.x, upper.y, lower.z), v) * factor.x)
					* factor.y) * (1 - factor.z)
			+ ((vs((uint3)(lower.x, lower.y, upper.z), v) * (1 - factor.x)
					+ vs((uint3)(upper.x, lower.y, upper.z), v) * factor.x)
					* (1 - factor.y)
					+ (vs((uint3)(lower.x, upper.y, upper.z), v)
							* (1 - factor.x)
							+ vs((uint3)(upper.x, upper.y, upper.z), v)
									* factor.x) * factor.y) * factor.z)
			* 0.00003051944088f;
}

inline float3 grad(float3 pos, const Volume v) {
	const float3 scaled_pos = (float3)((pos.x * v.size.x / v.dim.x) - 0.5f,
			(pos.y * v.size.y / v.dim.y) - 0.5f,
			(pos.z * v.size.z / v.dim.z) - 0.5f);
	const int3 base = (int3)(floor(scaled_pos.x), floor(scaled_pos.y),
			floor(scaled_pos.z));
	const float3 basef = (float3)(0);
	const float3 factor = (float3) fract(scaled_pos, (float3 *) &basef);
	const int3 lower_lower = max(base - (int3)(1), (int3)(0));
	const int3 lower_upper = max(base, (int3)(0));
	const int3 upper_lower = min(base + (int3)(1),
			convert_int3(v.size) - (int3)(1));
	const int3 upper_upper = min(base + (int3)(2),
			convert_int3(v.size) - (int3)(1));
	const int3 lower = lower_upper;
	const int3 upper = upper_lower;

	float3 gradient;

	gradient.x = (((vs((uint3)(upper_lower.x, lower.y, lower.z), v)
			- vs((uint3)(lower_lower.x, lower.y, lower.z), v)) * (1 - factor.x)
			+ (vs((uint3)(upper_upper.x, lower.y, lower.z), v)
					- vs((uint3)(lower_upper.x, lower.y, lower.z), v))
					* factor.x) * (1 - factor.y)
			+ ((vs((uint3)(upper_lower.x, upper.y, lower.z), v)
					- vs((uint3)(lower_lower.x, upper.y, lower.z), v))
					* (1 - factor.x)
					+ (vs((uint3)(upper_upper.x, upper.y, lower.z), v)
							- vs((uint3)(lower_upper.x, upper.y, lower.z), v))
							* factor.x) * factor.y) * (1 - factor.z)
			+ (((vs((uint3)(upper_lower.x, lower.y, upper.z), v)
					- vs((uint3)(lower_lower.x, lower.y, upper.z), v))
					* (1 - factor.x)
					+ (vs((uint3)(upper_upper.x, lower.y, upper.z), v)
							- vs((uint3)(lower_upper.x, lower.y, upper.z), v))
							* factor.x) * (1 - factor.y)
					+ ((vs((uint3)(upper_lower.x, upper.y, upper.z), v)
							- vs((uint3)(lower_lower.x, upper.y, upper.z), v))
							* (1 - factor.x)
							+ (vs((uint3)(upper_upper.x, upper.y, upper.z), v)
									- vs(
											(uint3)(lower_upper.x, upper.y,
													upper.z), v)) * factor.x)
							* factor.y) * factor.z;

	gradient.y = (((vs((uint3)(lower.x, upper_lower.y, lower.z), v)
			- vs((uint3)(lower.x, lower_lower.y, lower.z), v)) * (1 - factor.x)
			+ (vs((uint3)(upper.x, upper_lower.y, lower.z), v)
					- vs((uint3)(upper.x, lower_lower.y, lower.z), v))
					* factor.x) * (1 - factor.y)
			+ ((vs((uint3)(lower.x, upper_upper.y, lower.z), v)
					- vs((uint3)(lower.x, lower_upper.y, lower.z), v))
					* (1 - factor.x)
					+ (vs((uint3)(upper.x, upper_upper.y, lower.z), v)
							- vs((uint3)(upper.x, lower_upper.y, lower.z), v))
							* factor.x) * factor.y) * (1 - factor.z)
			+ (((vs((uint3)(lower.x, upper_lower.y, upper.z), v)
					- vs((uint3)(lower.x, lower_lower.y, upper.z), v))
					* (1 - factor.x)
					+ (vs((uint3)(upper.x, upper_lower.y, upper.z), v)
							- vs((uint3)(upper.x, lower_lower.y, upper.z), v))
							* factor.x) * (1 - factor.y)
					+ ((vs((uint3)(lower.x, upper_upper.y, upper.z), v)
							- vs((uint3)(lower.x, lower_upper.y, upper.z), v))
							* (1 - factor.x)
							+ (vs((uint3)(upper.x, upper_upper.y, upper.z), v)
									- vs(
											(uint3)(upper.x, lower_upper.y,
													upper.z), v)) * factor.x)
							* factor.y) * factor.z;

	gradient.z = (((vs((uint3)(lower.x, lower.y, upper_lower.z), v)
			- vs((uint3)(lower.x, lower.y, lower_lower.z), v)) * (1 - factor.x)
			+ (vs((uint3)(upper.x, lower.y, upper_lower.z), v)
					- vs((uint3)(upper.x, lower.y, lower_lower.z), v))
					* factor.x) * (1 - factor.y)
			+ ((vs((uint3)(lower.x, upper.y, upper_lower.z), v)
					- vs((uint3)(lower.x, upper.y, lower_lower.z), v))
					* (1 - factor.x)
					+ (vs((uint3)(upper.x, upper.y, upper_lower.z), v)
							- vs((uint3)(upper.x, upper.y, lower_lower.z), v))
							* factor.x) * factor.y) * (1 - factor.z)
			+ (((vs((uint3)(lower.x, lower.y, upper_upper.z), v)
					- vs((uint3)(lower.x, lower.y, lower_upper.z), v))
					* (1 - factor.x)
					+ (vs((uint3)(upper.x, lower.y, upper_upper.z), v)
							- vs((uint3)(upper.x, lower.y, lower_upper.z), v))
							* factor.x) * (1 - factor.y)
					+ ((vs((uint3)(lower.x, upper.y, upper_upper.z), v)
							- vs((uint3)(lower.x, upper.y, lower_upper.z), v))
							* (1 - factor.x)
							+ (vs((uint3)(upper.x, upper.y, upper_upper.z), v)
									- vs(
											(uint3)(upper.x, upper.y,
													lower_upper.z), v))
									* factor.x) * factor.y) * factor.z;

	return gradient
			* (float3)(v.dim.x / v.size.x, v.dim.y / v.size.y,
					v.dim.z / v.size.z) * (0.5f * 0.00003051944088f);
}

inline float3 get_translation(const Matrix4 view) {
	return (float3)(view.data[0].w, view.data[1].w, view.data[2].w);
}

inline float3 myrotate(const Matrix4 M, const float3 v) {
	return (float3)(dot((float3)(M.data[0].x, M.data[0].y, M.data[0].z), v),
			dot((float3)(M.data[1].x, M.data[1].y, M.data[1].z), v),
			dot((float3)(M.data[2].x, M.data[2].y, M.data[2].z), v));
}

float4 raycast(const Volume v, const uint2 pos, const Matrix4 view,
		const float nearPlane, const float farPlane, const float step,
		const float largestep) {

	const float3 origin = get_translation(view);
	const float3 direction = myrotate(view, (float3)(pos.x, pos.y, 1.f));

	// intersect ray with a box
	//
	// www.siggraph.org/education/materials/HyperGraph/raytrace/rtinter3.htm
	// compute intersection of ray with all six bbox planes
	const float3 invR = (float3)(1.0f) / direction;
	const float3 tbot = (float3) - 1 * invR * origin;
	const float3 ttop = invR * (v.dim - origin);

	// re-order intersections to find smallest and largest on each axis
	const float3 tmin = fmin(ttop, tbot);
	const float3 tmax = fmax(ttop, tbot);

	// find the largest tmin and the smallest tmax
	const float largest_tmin = fmax(fmax(tmin.x, tmin.y), fmax(tmin.x, tmin.z));
	const float smallest_tmax = fmin(fmin(tmax.x, tmax.y),
			fmin(tmax.x, tmax.z));

	// check against near and far plane
	const float tnear = fmax(largest_tmin, nearPlane);
	const float tfar = fmin(smallest_tmax, farPlane);

	if (tnear < tfar) {
		// first walk with largesteps until we found a hit
		float t = tnear;
		float stepsize = largestep;
		float f_t = interp(origin + direction * t, v);
		float f_tt = 0;
		if (f_t > 0) { // ups, if we were already in it, then don't render anything here
			for (; t < tfar; t += stepsize) {
				f_tt = interp(origin + direction * t, v);
				if (f_tt < 0)                  // got it, jump out of inner loop
					break;
				if (f_tt < 0.8f)               // coming closer, reduce stepsize
					stepsize = step;
				f_t = f_tt;
			}
			if (f_tt < 0) {           // got it, calculate accurate intersection
				t = t + stepsize * f_tt / (f_t - f_tt);
				return (float4)(origin + direction * t, t);
			}
		}
	}

	return (float4)(0);
}

__kernel void renderVolumeKernel( __global uchar * render,
		__global short2 * v_data,
		const uint3 v_size,
		const float3 v_dim,
		const float4 view0,
		const float4 view1,
		const float4 view2,
		const float4 view3,
		const float nearPlane,
		const float farPlane,
		const float step,
		const float largestep,
		const float3 light,
		const float3 ambient) {

	Matrix4 view;
	view.data[0] = view0;
	view.data[1] = view1;
	view.data[2] = view2;
	view.data[3] = view3;

	const Volume v = {v_size, v_dim,v_data};

	const uint2 pos = (uint2) (get_global_id(0),get_global_id(1));
	const int sizex = get_global_size(0);

	float4 hit = raycast( v, pos, view, nearPlane, farPlane,step, largestep);

	if(hit.w > 0) {
		const float3 test = as_float3(hit);
		float3 surfNorm = grad(test,v);
		if(length(surfNorm) > 0) {
			const float3 diff = normalize(light - test);
			const float dir = fmax(dot(normalize(surfNorm), diff), 0.f);
			const float3 col = clamp((float3)(dir) + ambient, 0.f, 1.f) * (float3) 255;
			vstore4((uchar4)(col.x, col.y, col.z, 0), pos.x + sizex * pos.y, render); // The forth value in uchar4 is padding for memory alignement and so it is for following uchar4 
		} else {
			vstore4((uchar4)(0, 0, 0, 0), pos.x + sizex * pos.y, render);
		}
	} else {
		vstore4((uchar4)(0, 0, 0, 0), pos.x + sizex * pos.y, render);
	}

}
