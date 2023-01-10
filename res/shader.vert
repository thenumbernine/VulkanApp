#version 450

// crashes ... 
#if 0
layout(binding = 0) uniform UniformBufferObject {
    mat4 model;
    mat4 view;
    mat4 proj;
} ubo;
#endif

layout(location = 0) in vec2 inPosition;
layout(location = 1) in vec3 inColor;

layout(location = 0) out vec3 fragColor;

void main() {
	gl_Position = 
#if 0		
		ubo.proj * ubo.view * ubo.model * 
#endif		
		vec4(inPosition, 0.0, 1.0);
	fragColor = inColor;
}
