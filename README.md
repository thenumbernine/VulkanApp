Making a minimal test case of Vulkan on SDL.
Maybe I'll turn it into a Vulkan parent class or object or something.
Oh what's that, there is no internal compiler into SPIR-V?  You have to use 3rd party tools?  Considering what a mess the consistence of any GLSL driver is across OS's even within the same GPU company, I can't say I'm surprised.
So now how in the world do I get my hands on glslc or clspv?
Hmm looks interesting: https://github.com/google/clspv

Using this:

- https://www.gamedev.net/forums/topic/699117-vulkan-with-sdl2-getting-started/
- https://vulkan-tutorial.com/Drawing_a_triangle/Setup/Instance
- https://docs.tizen.org/application/native/guides/graphics/vulkan/
- https://stackoverflow.com/questions/52388836/how-to-compile-opencl-kernels-to-spir-v-using-clang

maybe?
- `clang -cc1 -emit-spirv -triple=spir-unknown-unknown -cl-std=c++ -I include kernel.cl -o kernel.spv #For OpenCL C++`
- `clang -cc1 -emit-spirv -triple=spir-unknown-unknown -cl-std=CL2.0 -include opencl.h kernel.cl -o kernel.spv #For OpenCL C`
