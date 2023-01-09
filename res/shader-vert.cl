kernel void vert(
	global float4 const * const gl_Vertex,
	global float4 * const gl_Position,
	global float4 * const varyingColor
) {
	size_t const i = get_global_id(0);
	gl_Position[i] = gl_Vertex[i];
	varyingColor[i] = (float4){1., 0., 0., 1.};
}
