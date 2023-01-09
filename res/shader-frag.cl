kernel void frag(
	global float4 const * const varyingColor,
	global float4 * const gl_FragColor
) {
	size_t const i = get_global_id(0);
	gl_FragColor[i] = varyingColor[i];
}
