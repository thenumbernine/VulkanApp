#include "SDLApp/SDLApp.h"
#include "Image/Image.h"
#include "Common/Exception.h"
#include "Common/Macros.h"	//LINE_STRING
#include "Common/File.h"
#include "Common/Function.h"
#include "Tensor/Tensor.h"
#include <SDL_vulkan.h>
#include <vulkan/vulkan.hpp>
#include <iostream>	//debugging only
#include <set>
#include <chrono>

#define NAME_PAIR(x)	#x, x

template<
	typename F,
	typename... Args
> void vulkanSafe(
	std::string what,
	F f,
	Args&&... args
) {
	VkResult res = f(std::forward<Args>(args)...);
	if (res != VK_SUCCESS) {
		throw Common::Exception() << what << " failed: " << res;
	}
}

#define VULKAN_SAFE(f, ...) 	vulkanSafe(FILE_AND_LINE " " #f, f, __VA_ARGS__)

#define SDL_VULKAN_SAFE(f, ...) {\
	if (f(__VA_ARGS__) == SDL_FALSE) {\
		throw Common::Exception() << FILE_AND_LINE " " #f " failed: " << SDL_GetError();\
	}\
}

template<
	typename T,
	typename F,
	typename... Args
>
auto vulkanEnum(
	std::string what,
	F f,
	Args&&... args
) {
	if constexpr (std::is_same_v<
		typename Common::FunctionPointer<F>::Return,
		void
	>) {
		uint32_t count = {};
		std::vector<T> result;
		f(std::forward<Args>(args)..., &count, nullptr);
		result.resize(count);
		if (count) {
			f(std::forward<Args>(args)..., &count, result.data());
		}
		return result;
	} else if constexpr (std::is_same_v<
		typename Common::FunctionPointer<F>::Return,
		VkResult
	>) {
		uint32_t count = {};
		std::vector<T> result;
		vulkanSafe(what, f, std::forward<Args>(args)..., &count, nullptr);
		result.resize(count);
		if (count) {
			vulkanSafe(what, f, std::forward<Args>(args)..., &count, result.data());
		}
		return result;
	} else {
		//static_assert(false, "I don't know how to handle this");
		throw Common::Exception() << "I don't know how to handle this";
	}
}

#define VULKAN_ENUM_SAFE(T, f, ...) (vulkanEnum<T>(std::string(FILE_AND_LINE " " #f, (f), __VA_ARGS__))

template<typename real>
real degToRad(real x) {
	return x * (real)(M_PI / 180.);
}

// TODO put this somewhere maybe

namespace Tensor {

//glRotatef
template<typename real, typename Src>
requires (Src::dims() == int2(4,4))
_mat<real,4,4> rotate(
	Src src,
	real rad,
	_vec<real,3> axis
) {
	auto q = Tensor::_quat<real>(axis.x, axis.y, axis.z, rad)
		.fromAngleAxis();
	auto x = q.xAxis();
	auto y = q.yAxis();
	auto z = q.zAxis();
	//which is faster? 
	// this 4x4 mat mul?
	// or quat-rotate the col vectors of mq?
	_mat<real,4,4> mq = {
		{x.x, y.x, z.x, 0},
		{x.y, y.y, z.y, 0},
		{x.z, y.z, z.z, 0},
		{0, 0, 0, 1}
	};
	return src * mq;
}

//gluLookAt
template<typename real>
_mat<real,4,4> lookAt(
	_vec<real,3> eye,
	_vec<real,3> center,
	_vec<real,3> up
) {
	auto l = (center - eye).normalize();
	auto s = l.cross(up).normalize();
	auto up2 = s.cross(l);
	return _mat<real,4,4>{
		{s.x, up2.x, -l.x, -eye.x},
		{s.y, up2.y, -l.y, -eye.y},
		{s.z, up2.z, -l.z, -eye.z},
		{0, 0, 0, 1},
	};
}

//gluPerspective
template<typename real>
_mat<real,4,4> perspective(
	real fovy,
	real aspectRatio,
	real zNear,
	real zFar
) {
	real f = 1./tan(fovy*(real).5);
	real neginvdz = (real)1 / (zNear - zFar);
	return _mat<real,4,4>{
		{f/aspectRatio, 0, 0, 0},
		{0, f, 0, 0},
		{0, 0, (zFar+zNear) * neginvdz, (real)2*zFar*zNear * neginvdz},
		{0, 0, -1, 0},
	};
}

}

// why do I think there are already similar classes in vulkan.hpp?

struct Vertex {
	Tensor::float2 pos;
	Tensor::float3 color;

	static auto getBindingDescription() {
		return VkVertexInputBindingDescription{
			0,								//binding
			sizeof(Vertex),					//stride
			VK_VERTEX_INPUT_RATE_VERTEX,	//inputRate
		};
	}

	// TODO instead put fields = tuple of member refs
	// then have a method for 'fields-to-binding-descriptions'
	// and a method for 'fields-to-attribute-descriptions'
	static auto getAttributeDescriptions() {
		return std::array<VkVertexInputAttributeDescription, 2>{
			VkVertexInputAttributeDescription{
				0,							//location
				0,							//binding
				VK_FORMAT_R32G32_SFLOAT,	//format
				offsetof(Vertex, pos),		//offset
			},
			VkVertexInputAttributeDescription{
				1,							//location
				0,							//binding
				VK_FORMAT_R32G32B32_SFLOAT,	//format
				offsetof(Vertex, color),	//offset
			},
		};
	}
};

struct UniformBufferObject {
	alignas(16) Tensor::float4x4 model;
	alignas(16) Tensor::float4x4 view;
	alignas(16) Tensor::float4x4 proj;
};

std::vector<Vertex> const vertices = {
	{{-0.5f, -0.5f}, {1.f, 0.f, 0.f}},
	{{0.5f, -0.5f}, {0.f, 1.f, 0.f}},
	{{0.5f, 0.5f}, {0.f, 0.f, 1.f}},
	{{-0.5f, 0.5f}, {1.f, 1.f, 1.f}}
};

std::vector<uint16_t> const indices = {
	0, 1, 2, 2, 3, 0
};

// hmm is it worth it to pass the child in, so that Traits can access child members, and use them for its methods?
// or at this point do I just leave the function calls in the child class?
template<typename Handle, typename Child>
struct VulkanTraits;


// null allocator.  this is all compile-resolved, right?
// it's not going to need extra space if I inherit from it right?
// but it will if I add it as a member?
// or will it not add space if I only make the members static methods that use 'this' ?  same same?
struct VulkanNullAllocator {
	VkAllocationCallbacks * getAllocator() { return nullptr; }
};

// custom allocator
struct VulkanAllocator {
	VkAllocationCallbacks * allocator = nullptr;
	VkAllocationCallbacks * getAllocator() { return allocator; }
};

template<
	typename Handle_,
	typename Allocator = VulkanNullAllocator
>
struct VulkanHandle : public Allocator {
	using Handle = Handle_;
protected:
	Handle handle = {};
public:
	auto const & operator()() const { return handle; }

	VulkanHandle() {} 
	VulkanHandle(Handle handle_) : handle(handle_) {} 
};


struct VulkanInstance : public VulkanHandle<VkInstance> {
	~VulkanInstance() {
		if (handle) vkDestroyInstance(handle, getAllocator());
	}
	
	PFN_vkVoidFunction getProcAddr(char const * const name) const {
		return vkGetInstanceProcAddr(handle, name);
	}

	std::vector<VkPhysicalDevice> getPhysicalDevices() const {
		return vulkanEnum<VkPhysicalDevice>(
			NAME_PAIR(vkEnumeratePhysicalDevices),
			handle
		);
	}
	
	// ************** from here on down, app-specific **************  

	// this does result in vkCreateInstance, 
	//  but the way it gest there is very application-specific
	VulkanInstance(
		::SDLApp::SDLApp const * const app,
		bool const enableValidationLayers
	) {
		// debug output

		{
#if 1
			auto availableLayers = vulkanEnum<VkLayerProperties>(
				NAME_PAIR(vkEnumerateInstanceLayerProperties)
			);
#else
#define EMPTY			
			std::vector<VkLayerProperties> availableLayers = VULKAN_ENUM_SAFE(
				VkLayerProperties,
				vkEnumerateInstanceLayerProperties,
				std::make_tuple()
			);
#endif
			std::cout << "vulkan layers:" << std::endl;
			for (auto const & layer : availableLayers) {
				std::cout << "\t" << layer.layerName << std::endl;
			}
		}

		// VkApplicationInfo needs title:
		auto title = app->getTitle();
		
		// vkCreateInstance needs appInfo
		VkApplicationInfo appInfo = {};
		appInfo.sType = VK_STRUCTURE_TYPE_APPLICATION_INFO;
		appInfo.pApplicationName = title.c_str();
		appInfo.applicationVersion = VK_MAKE_VERSION(1, 0, 0);
		appInfo.pEngineName = "No Engine";
		appInfo.engineVersion = VK_MAKE_VERSION(1, 0, 0);
		appInfo.apiVersion = VK_API_VERSION_1_0;

		// vkCreateInstance needs layerNames
		std::vector<char const *> layerNames;
		if (enableValidationLayers) {
			//insert which of those into our layerName for creating something or something
			//layerNames.push_back("VK_LAYER_LUNARG_standard_validation");	//nope
			layerNames.push_back("VK_LAYER_KHRONOS_validation");	//nope
		}
		
		// vkCreateInstance needs extensions
		auto extensions = getRequiredExtensions(app, enableValidationLayers);

		VkInstanceCreateInfo createInfo = {};
		createInfo.sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO;
		createInfo.pApplicationInfo = &appInfo;
		createInfo.enabledLayerCount = layerNames.size();
		createInfo.ppEnabledLayerNames = layerNames.data();
		createInfo.enabledExtensionCount = extensions.size();
		createInfo.ppEnabledExtensionNames = extensions.data();
		VULKAN_SAFE(vkCreateInstance, &createInfo, nullptr, &handle);
	}
protected:
	std::vector<char const *> getRequiredExtensions(
		::SDLApp::SDLApp const * const app,
		bool const enableValidationLayers
	) {
		uint32_t extensionCount = {};
		SDL_VULKAN_SAFE(SDL_Vulkan_GetInstanceExtensions, app->getWindow(), &extensionCount, nullptr);
		std::vector<const char *> extensions(extensionCount);
		SDL_VULKAN_SAFE(SDL_Vulkan_GetInstanceExtensions, app->getWindow(), &extensionCount, extensions.data());

		//debugging:
		std::cout << "vulkan extensions:" << std::endl;
		for (auto const & ext : extensions) {
			std::cout << "\t" << ext << std::endl;
		}

		if (enableValidationLayers) {
			extensions.push_back(VK_EXT_DEBUG_UTILS_EXTENSION_NAME);
		}

		return extensions;
	}
};

struct VulkanDebugMessenger : public VulkanHandle<VkDebugUtilsMessengerEXT> {
protected:
	VulkanInstance const * const instance = {};	//needed for getProcAddr and for handle in dtor
	
	PFN_vkCreateDebugUtilsMessengerEXT vkCreateDebugUtilsMessengerEXT = {};
	PFN_vkDestroyDebugUtilsMessengerEXT vkDestroyDebugUtilsMessengerEXT = {};

	static constexpr auto exts = std::make_tuple(
		std::make_pair("vkCreateDebugUtilsMessengerEXT", &VulkanDebugMessenger::vkCreateDebugUtilsMessengerEXT),
		std::make_pair("vkDestroyDebugUtilsMessengerEXT", &VulkanDebugMessenger::vkDestroyDebugUtilsMessengerEXT)
	);

public:
	~VulkanDebugMessenger() {
		// call destroy function
		if (vkDestroyDebugUtilsMessengerEXT && handle) {
			vkDestroyDebugUtilsMessengerEXT((*instance)(), handle, getAllocator());
		}
	}

	VulkanDebugMessenger(
		VulkanInstance const * const instance_
	) : instance(instance_) {
		Common::TupleForEach(exts, [this](auto x, size_t i) constexpr -> bool {
			auto name = std::get<0>(x);
			auto field = std::get<1>(x);
			this->*field = (std::decay_t<decltype(this->*field)>)instance->getProcAddr(name);
			if (!(this->*field)) {
				throw Common::Exception() << "vkGetInstanceProcAddr " << name << " failed";
			}
		});

		// call create function
		
		VkDebugUtilsMessengerCreateInfoEXT createInfo = {};
		createInfo.sType = VK_STRUCTURE_TYPE_DEBUG_UTILS_MESSENGER_CREATE_INFO_EXT;
		createInfo.messageSeverity = VK_DEBUG_UTILS_MESSAGE_SEVERITY_VERBOSE_BIT_EXT | VK_DEBUG_UTILS_MESSAGE_SEVERITY_WARNING_BIT_EXT | VK_DEBUG_UTILS_MESSAGE_SEVERITY_ERROR_BIT_EXT;
		createInfo.messageType = VK_DEBUG_UTILS_MESSAGE_TYPE_GENERAL_BIT_EXT | VK_DEBUG_UTILS_MESSAGE_TYPE_VALIDATION_BIT_EXT | VK_DEBUG_UTILS_MESSAGE_TYPE_PERFORMANCE_BIT_EXT;
		createInfo.pfnUserCallback = debugCallback;
		VULKAN_SAFE(vkCreateDebugUtilsMessengerEXT, (*instance)(), &createInfo, nullptr, &handle);
	}

// app-specific callback
protected:
	static VKAPI_ATTR VkBool32 VKAPI_CALL debugCallback(VkDebugUtilsMessageSeverityFlagBitsEXT messageSeverity, VkDebugUtilsMessageTypeFlagsEXT messageType, const VkDebugUtilsMessengerCallbackDataEXT* pCallbackData, void* pUserData) {
		std::cerr << "validation layer: " << pCallbackData->pMessage << std::endl;
		return VK_FALSE;
	}
};

struct VulkanSurface : public VulkanHandle<VkSurfaceKHR> {
protected:
	VkInstance instance = {};	//from VulkanCommon, needs to be held for dtor to work
public:
	~VulkanSurface() {
		if (handle) vkDestroySurfaceKHR(instance, handle, getAllocator());
	}

	VulkanSurface(
		SDL_Window * const window,
		VkInstance instance_
	) : instance(instance_) {
		// https://vulkan-tutorial.com/Drawing_a_triangle/Presentation/Window_surface
		SDL_VULKAN_SAFE(SDL_Vulkan_CreateSurface, window, instance, &handle);
	}
};

struct VulkanPhysicalDevice : public VulkanHandle<VkPhysicalDevice> {
	VulkanPhysicalDevice(VkPhysicalDevice handle_)
	: VulkanHandle<VkPhysicalDevice>(handle_) 
	{}

	VkPhysicalDeviceProperties getProperties() const {
		VkPhysicalDeviceProperties physDevProps;
		vkGetPhysicalDeviceProperties(handle, &physDevProps);
		return physDevProps;
	}

	VkPhysicalDeviceMemoryProperties getMemoryProperties() const {
		VkPhysicalDeviceMemoryProperties memProps = {};
		vkGetPhysicalDeviceMemoryProperties(handle, &memProps);
		return memProps;
	}
	
	std::vector<VkQueueFamilyProperties> getQueueFamilyProperties() const {
		return vulkanEnum<VkQueueFamilyProperties>(
			NAME_PAIR(vkGetPhysicalDeviceQueueFamilyProperties),
			handle
		);
	}

	std::vector<VkExtensionProperties> getExtensionProperties(
		char const * const layerName = nullptr
	) const {
		return vulkanEnum<VkExtensionProperties>(
			NAME_PAIR(vkEnumerateDeviceExtensionProperties),
			handle,
			layerName
		);
	}

	bool getSurfaceSupport(
		uint32_t queueFamilyIndex,
		VkSurfaceKHR surface
	) const {
		VkBool32 presentSupport = VK_FALSE;
		VULKAN_SAFE(vkGetPhysicalDeviceSurfaceSupportKHR, handle, queueFamilyIndex, surface, &presentSupport);
		return !!presentSupport;
	}

	//pass-by-value ok?
	// should these be physical-device-specific or surface-specific?
	//  if a surface needs a physical device ... the latter?
	VkSurfaceCapabilitiesKHR getSurfaceCapabilities(
		VkSurfaceKHR surface
	) const {
		VkSurfaceCapabilitiesKHR caps = {};
		VULKAN_SAFE(vkGetPhysicalDeviceSurfaceCapabilitiesKHR, handle, surface, &caps);
		return caps;
	}

	std::vector<VkSurfaceFormatKHR> getSurfaceFormats(
		VkSurfaceKHR surface
	) const {
		return vulkanEnum<VkSurfaceFormatKHR>(
			NAME_PAIR(vkGetPhysicalDeviceSurfaceFormatsKHR),
			handle,
			surface
		);
	}

	std::vector<VkPresentModeKHR> getSurfacePresentModes(
		VkSurfaceKHR surface
	) const {
		return vulkanEnum<VkPresentModeKHR>(
			NAME_PAIR(vkGetPhysicalDeviceSurfacePresentModesKHR),
			handle,
			surface
		);
	}

	VkDevice createDevice(
		VkDeviceCreateInfo const * const createInfo,
		VkAllocationCallbacks const * const allocator = nullptr
	) const {
		VkDevice device = {};
		VULKAN_SAFE(vkCreateDevice, handle, createInfo, allocator, &device);
		return device;
	}

	// this is halfway app-specific.  it was in the tut's organization but I'm 50/50 if it is good design

	uint32_t findMemoryType(
		uint32_t typeFilter,
		VkMemoryPropertyFlags props
	) const {
		VkPhysicalDeviceMemoryProperties memProps = getMemoryProperties();
		for (uint32_t i = 0; i < memProps.memoryTypeCount; i++) {
			if ((typeFilter & (1 << i)) && (memProps.memoryTypes[i].propertyFlags & props) == props) {
				return i;
			}
		}
		throw Common::Exception() << ("failed to find suitable memory type!");
	}

	struct SwapChainSupportDetails {
		VkSurfaceCapabilitiesKHR capabilities;
		std::vector<VkSurfaceFormatKHR> formats;
		std::vector<VkPresentModeKHR> presentModes;
	};

	SwapChainSupportDetails querySwapChainSupport(
		VkSurfaceKHR surface
	) const {
		return SwapChainSupportDetails(
			getSurfaceCapabilities(surface),	//capabilities
			getSurfaceFormats(surface),			//formats
			getSurfacePresentModes(surface)		//presentModes
		);
	}

	// ************** from here on down, app-specific **************  

	// used by the application for specific physical device querying (should be a subclass of the general VulkanPhysicalDevice)
	VulkanPhysicalDevice(
		VulkanInstance const * const instance,
		VkSurfaceKHR surface,								// needed by isDeviceSuitable -> findQueueFamilie
		std::vector<char const *> const & deviceExtensions	// needed by isDeviceSuitable -> checkDeviceExtensionSupport
	) {
		// TODO return handle or class wrapper?
		auto physDevs = instance->getPhysicalDevices();
		//debug:
		std::cout << "devices:" << std::endl;
		for (auto const & h : physDevs) {
			auto o = VulkanPhysicalDevice(h);
			auto props = o.getProperties();
			std::cout
				<< "\t"
				<< props.deviceName
				<< " type=" << props.deviceType
				<< std::endl;
		}

		for (auto const & h : physDevs) {
			auto o = VulkanPhysicalDevice(h);
			if (o.isDeviceSuitable(surface, deviceExtensions)) {
				handle = h;
				break;
			}
		}

		if (!handle) throw Common::Exception() << "failed to find a suitable GPU!";
	}

protected:
	bool isDeviceSuitable(
		VkSurfaceKHR surface,								// needed by findQueueFamilies, querySwapChainSupport
		std::vector<char const *> const & deviceExtensions	// needed by checkDeviceExtensionSupport
	) const {
#if 0	// i'm not seeing queue families indices and the actual physicalDevice info query overlap
		// or is querying individual devices properties not a thing anymore?
		// do you just search for the queue family bit? graphics? compute? whatever?

		auto physDevProps = getProperties();
		VkPhysicalDeviceFeatures deviceFeatures;
		vkGetPhysicalDeviceFeatures(handle, &deviceFeatures);
		// TODO sort by score and pick the best
		return physDevProps.deviceType == VK_PHYSICAL_DEVICE_TYPE_DISCRETE_GPU
			|| physDevProps.deviceType == VK_PHYSICAL_DEVICE_TYPE_INTEGRATED_GPU
			|| physDevProps.deviceType == VK_PHYSICAL_DEVICE_TYPE_VIRTUAL_GPU
		;
			// && deviceFeatures.geometryShader;
#endif
		auto indices = findQueueFamilies(surface);
		
		bool extensionsSupported = checkDeviceExtensionSupport(deviceExtensions);

		bool swapChainAdequate = false;
		if (extensionsSupported) {
			auto swapChainSupport = querySwapChainSupport(surface);
			swapChainAdequate = !swapChainSupport.formats.empty() && !swapChainSupport.presentModes.empty();
		}

		return indices.isComplete()
			&& extensionsSupported
			&& swapChainAdequate;
	}

	//used by isDeviceSuitable
	bool checkDeviceExtensionSupport(
		std::vector<char const *> const & deviceExtensions
	) const {
		std::set<std::string> requiredExtensions(deviceExtensions.begin(), deviceExtensions.end());
		for (auto const & extension : getExtensionProperties()) {
			requiredExtensions.erase(extension.extensionName);
		}
		return requiredExtensions.empty();
	}

public:
	struct QueueFamilyIndices {
		std::optional<uint32_t> graphicsFamily;
		std::optional<uint32_t> presentFamily;
		bool isComplete() {
			return graphicsFamily.has_value()
				&& presentFamily.has_value();
		}
	};

	// used by a few functions
	// needs surface
	QueueFamilyIndices findQueueFamilies(
		VkSurfaceKHR surface
	) const {
		QueueFamilyIndices indices;
		auto queueFamilies = getQueueFamilyProperties();
		for (uint32_t i = 0; i < queueFamilies.size(); ++i) {
			auto const & f = queueFamilies[i];
			if (f.queueFlags & VK_QUEUE_GRAPHICS_BIT) {
				indices.graphicsFamily = i;
			}
			if (getSurfaceSupport(i, surface)) {
				indices.presentFamily = i;
			}
			if (indices.isComplete()) {
				return indices;
			}
		}
		throw Common::Exception() << "couldn't find all indices";
	}
};

// validationLayers matches in checkValidationLayerSupport and initLogicalDevice
std::vector<char const *> validationLayers = {
	"VK_LAYER_KHRONOS_validation"
};

struct VulkanLogicalDevice : public VulkanHandle<VkDevice> {
public:
	VkQueue graphicsQueue = {};
	VkQueue presentQueue = {};
public:
	
	~VulkanLogicalDevice() {
		if (handle) vkDestroyDevice(handle, getAllocator());
	}
	
	VkQueue getQueue(
		uint32_t queueFamilyIndex,
		uint32_t queueIndex = 0
	) const {
		VkQueue result;
		vkGetDeviceQueue(handle, queueFamilyIndex, queueIndex, &result);
		return result;
	}
	
	void waitIdle() const {
		VULKAN_SAFE(vkDeviceWaitIdle, handle);
	}

	// maybe there's no need for these 'create' functions 

#define CREATE_CREATER(name, suffix)\
	Vk##name##suffix create##name(\
		Vk##name##CreateInfo##suffix const * const createInfo,\
		VkAllocationCallbacks const * const allocator = nullptr\
	) const {\
		Vk##name##suffix result = {};\
		VULKAN_SAFE(vkCreate##name##suffix, handle, createInfo, allocator, &result);\
		return result;\
	}

CREATE_CREATER(Swapchain, KHR)
CREATE_CREATER(RenderPass, )
CREATE_CREATER(ImageView, )
CREATE_CREATER(Framebuffer, )
CREATE_CREATER(DescriptorSetLayout, )
CREATE_CREATER(ShaderModule, )
CREATE_CREATER(PipelineLayout, )
CREATE_CREATER(CommandPool, )
CREATE_CREATER(Buffer, )

	VkPipeline createGraphicsPipelines(
		VkPipelineCache pipelineCache,
		size_t numCreateInfo,
		VkGraphicsPipelineCreateInfo const * const createInfo,
		VkAllocationCallbacks const * const allocator = nullptr
	) const {
		VkPipeline result = {};
		VULKAN_SAFE(vkCreateGraphicsPipelines, handle, pipelineCache, numCreateInfo, createInfo, allocator, &result);
		return result;
	}

	// ************** from here on down, app-specific **************  
	
	VulkanLogicalDevice(
		VulkanPhysicalDevice const * const physicalDevice,
		VkSurfaceKHR surface,
		std::vector<char const *> const & deviceExtensions,
		bool enableValidationLayers
	) {
		auto indices = physicalDevice->findQueueFamilies(surface);

		float queuePriority = 1;
		std::vector<VkDeviceQueueCreateInfo> queueCreateInfos;
		for (uint32_t queueFamily : std::set<uint32_t>{
			indices.graphicsFamily.value(),
			indices.presentFamily.value(),
		}) {
			VkDeviceQueueCreateInfo queueCreateInfo = {};
			queueCreateInfo.sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO;
			queueCreateInfo.queueFamilyIndex = queueFamily;
			queueCreateInfo.queueCount = 1;
			queueCreateInfo.pQueuePriorities = &queuePriority;
			queueCreateInfos.push_back(queueCreateInfo);
		}
		
		VkPhysicalDeviceFeatures deviceFeatures = {}; // empty

		VkDeviceCreateInfo createInfo = {};
		createInfo.sType = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO;
		createInfo.queueCreateInfoCount = static_cast<uint32_t>(queueCreateInfos.size());
		createInfo.pQueueCreateInfos = queueCreateInfos.data();
		createInfo.pEnabledFeatures = &deviceFeatures;
		createInfo.enabledExtensionCount = static_cast<uint32_t>(deviceExtensions.size());
		createInfo.ppEnabledExtensionNames = deviceExtensions.data();
		if (enableValidationLayers) {
			createInfo.enabledLayerCount = static_cast<uint32_t>(validationLayers.size());
			createInfo.ppEnabledLayerNames = validationLayers.data();
		} else {
			createInfo.enabledLayerCount = 0;
		}
		handle = physicalDevice->createDevice(&createInfo);
	
		graphicsQueue = getQueue(indices.graphicsFamily.value());
		presentQueue = getQueue(indices.presentFamily.value());
	}
};

struct VulkanRenderPass : public VulkanHandle<VkRenderPass> {
protected:
	//held
	VkDevice device = {};
public:
	~VulkanRenderPass() {
		if (handle) vkDestroyRenderPass(device, handle, getAllocator());
	}
	
	// ************** from here on down, app-specific **************  

	VulkanRenderPass(
		VulkanLogicalDevice const * const device_,
		VkFormat swapChainImageFormat
	) : device((*device_)()) {
		VkAttachmentDescription colorAttachment = {};
		colorAttachment.format = swapChainImageFormat;
		colorAttachment.samples = VK_SAMPLE_COUNT_1_BIT;
		colorAttachment.loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR;
		colorAttachment.storeOp = VK_ATTACHMENT_STORE_OP_STORE;
		colorAttachment.stencilLoadOp = VK_ATTACHMENT_LOAD_OP_DONT_CARE;
		colorAttachment.stencilStoreOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;
		colorAttachment.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
		colorAttachment.finalLayout = VK_IMAGE_LAYOUT_PRESENT_SRC_KHR;

		VkAttachmentReference colorAttachmentRef = {};
		colorAttachmentRef.attachment = 0;
		colorAttachmentRef.layout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;

		VkSubpassDescription subpass = {};
		subpass.pipelineBindPoint = VK_PIPELINE_BIND_POINT_GRAPHICS;
		subpass.colorAttachmentCount = 1;
		subpass.pColorAttachments = &colorAttachmentRef;

		VkSubpassDependency dependency = {};
		dependency.srcSubpass = VK_SUBPASS_EXTERNAL;
		dependency.dstSubpass = 0;
		dependency.srcStageMask = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT;
		dependency.srcAccessMask = 0;
		dependency.dstStageMask = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT;
		dependency.dstAccessMask = VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT;

		VkRenderPassCreateInfo renderPassInfo = {};
		renderPassInfo.sType = VK_STRUCTURE_TYPE_RENDER_PASS_CREATE_INFO;
		renderPassInfo.attachmentCount = 1;
		renderPassInfo.pAttachments = &colorAttachment;
		renderPassInfo.subpassCount = 1;
		renderPassInfo.pSubpasses = &subpass;
		renderPassInfo.dependencyCount = 1;
		renderPassInfo.pDependencies = &dependency;
		handle = device_->createRenderPass(&renderPassInfo);
	}
};

struct VulkanSwapChain : public VulkanHandle<VkSwapchainKHR> {
protected:
	//owned
	std::unique_ptr<VulkanRenderPass> renderPass;
	// hold for this class lifespan
	VkDevice device;
public:
	VkExtent2D extent;
	
	// I would combine these into one struct so they can be dtored together
	// but it seems vulkan wants VkImages linear for its getter?
	std::vector<VkImage> images;
	std::vector<VkImageView> imageViews;
	std::vector<VkFramebuffer> framebuffers;

public:
	VkRenderPass getRenderPass() const { return (*renderPass)(); }

	~VulkanSwapChain() {
		for (auto framebuffer : framebuffers) {
			vkDestroyFramebuffer(device, framebuffer, getAllocator());
		}
		renderPass = nullptr;
		for (auto imageView : imageViews) {
			vkDestroyImageView(device, imageView, getAllocator());
		}
		if (handle) vkDestroySwapchainKHR(device, handle, getAllocator());
	}

	// should this be a 'devices' or a 'swapchain' method?
	std::vector<VkImage> getImages() const {
		return vulkanEnum<VkImage>(NAME_PAIR(vkGetSwapchainImagesKHR), device, handle);
	}
	
	// ************** from here on down, app-specific **************  
	
	VulkanSwapChain(
		Tensor::int2 screenSize,
		VulkanPhysicalDevice * const physicalDevice,
		VulkanLogicalDevice const * const device_,
		VkSurfaceKHR surface
	) : device((*device_)()) {
		auto swapChainSupport = physicalDevice->querySwapChainSupport(surface);

		VkSurfaceFormatKHR surfaceFormat = chooseSwapSurfaceFormat(swapChainSupport.formats);
		VkPresentModeKHR presentMode = chooseSwapPresentMode(swapChainSupport.presentModes);
		extent = chooseSwapExtent(screenSize, swapChainSupport.capabilities);

		// how come imageCount is one less than vkGetSwapchainImagesKHR gives?
		// maxImageCount == 0 means no max?
		uint32_t imageCount = swapChainSupport.capabilities.minImageCount + 1;
		if (swapChainSupport.capabilities.maxImageCount > 0) {
			imageCount = std::min(imageCount, swapChainSupport.capabilities.maxImageCount);
		}

		VkSwapchainCreateInfoKHR createInfo = {};
		createInfo.sType = VK_STRUCTURE_TYPE_SWAPCHAIN_CREATE_INFO_KHR;
		createInfo.surface = surface;
		createInfo.minImageCount = imageCount;
		createInfo.imageFormat = surfaceFormat.format;
		createInfo.imageColorSpace = surfaceFormat.colorSpace;
		createInfo.imageExtent = extent;
		createInfo.imageArrayLayers = 1;
		createInfo.imageUsage = VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT;

		auto indices = physicalDevice->findQueueFamilies(surface);
		uint32_t queueFamilyIndices[] = {indices.graphicsFamily.value(), indices.presentFamily.value()};
		if (indices.graphicsFamily != indices.presentFamily) {
			createInfo.imageSharingMode = VK_SHARING_MODE_CONCURRENT;
			createInfo.queueFamilyIndexCount = 2;
			createInfo.pQueueFamilyIndices = queueFamilyIndices;
		} else {
			createInfo.imageSharingMode = VK_SHARING_MODE_EXCLUSIVE;
		}

		createInfo.preTransform = swapChainSupport.capabilities.currentTransform;
		createInfo.compositeAlpha = VK_COMPOSITE_ALPHA_OPAQUE_BIT_KHR;
		createInfo.presentMode = presentMode;
		createInfo.clipped = VK_TRUE;
		handle = device_->createSwapchain(&createInfo);

		images = getImages();
		imageViews.resize(images.size());
		for (size_t i = 0; i < images.size(); i++) {
			VkImageViewCreateInfo createInfo = {};
			createInfo.sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO;
			createInfo.image = images[i];
			createInfo.viewType = VK_IMAGE_VIEW_TYPE_2D;
			createInfo.format = surfaceFormat.format;
			createInfo.components.r = VK_COMPONENT_SWIZZLE_IDENTITY;
			createInfo.components.g = VK_COMPONENT_SWIZZLE_IDENTITY;
			createInfo.components.b = VK_COMPONENT_SWIZZLE_IDENTITY;
			createInfo.components.a = VK_COMPONENT_SWIZZLE_IDENTITY;
			createInfo.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
			createInfo.subresourceRange.baseMipLevel = 0;
			createInfo.subresourceRange.levelCount = 1;
			createInfo.subresourceRange.baseArrayLayer = 0;
			createInfo.subresourceRange.layerCount = 1;
			imageViews[i] = device_->createImageView(&createInfo);
		}
	
		renderPass = std::make_unique<VulkanRenderPass>(device_, surfaceFormat.format);
		framebuffers.resize(imageViews.size());
		for (size_t i = 0; i < imageViews.size(); i++) {
			VkFramebufferCreateInfo framebufferInfo = {};
			framebufferInfo.sType = VK_STRUCTURE_TYPE_FRAMEBUFFER_CREATE_INFO;
			framebufferInfo.renderPass = (*renderPass)();
			framebufferInfo.attachmentCount = 1;
			framebufferInfo.pAttachments = &imageViews[i];
			framebufferInfo.width = extent.width;
			framebufferInfo.height = extent.height;
			framebufferInfo.layers = 1;
			framebuffers[i] = device_->createFramebuffer(&framebufferInfo);
		}
	}

protected:
	static VkSurfaceFormatKHR chooseSwapSurfaceFormat(
		const std::vector<VkSurfaceFormatKHR>& availableFormats
	) {
		for (auto const & availableFormat : availableFormats) {
			if (availableFormat.format == VK_FORMAT_B8G8R8A8_SRGB &&
				availableFormat.colorSpace == VK_COLOR_SPACE_SRGB_NONLINEAR_KHR
			) {
				return availableFormat;
			}
		}
		return availableFormats[0];
	}

	static VkPresentModeKHR chooseSwapPresentMode(
		const std::vector<VkPresentModeKHR>& availablePresentModes
	) {
		for (auto const & availablePresentMode : availablePresentModes) {
			if (availablePresentMode == VK_PRESENT_MODE_MAILBOX_KHR) {
				return availablePresentMode;
			}
		}
		return VK_PRESENT_MODE_FIFO_KHR;
	}

	static VkExtent2D chooseSwapExtent(
		Tensor::int2 screenSize,
		VkSurfaceCapabilitiesKHR const & capabilities
	) {
		if (capabilities.currentExtent.width != std::numeric_limits<uint32_t>::max()) {
			return capabilities.currentExtent;
		} else {
			VkExtent2D actualExtent = {
				static_cast<uint32_t>(screenSize.x),
				static_cast<uint32_t>(screenSize.y)
			};
			actualExtent.width = std::clamp(actualExtent.width, capabilities.minImageExtent.width, capabilities.maxImageExtent.width);
			actualExtent.height = std::clamp(actualExtent.height, capabilities.minImageExtent.height, capabilities.maxImageExtent.height);
			return actualExtent;
		}
	}
};

//only used by VulkanGraphicsPipeline
struct VulkanDescriptorSetLayout : public VulkanHandle<VkDescriptorSetLayout> {
protected:
	//held for dtor
	VkDevice device = {};
public:
	~VulkanDescriptorSetLayout() {
		if (handle) {
			vkDestroyDescriptorSetLayout(device, handle, getAllocator());
		}
	}
	
	VulkanDescriptorSetLayout(
		VulkanLogicalDevice const * const device_
	) : device((*device_)()) {
		VkDescriptorSetLayoutBinding uboLayoutBinding = {};
		uboLayoutBinding.binding = 0;
		uboLayoutBinding.descriptorCount = 1;
		uboLayoutBinding.descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
		uboLayoutBinding.pImmutableSamplers = nullptr;
		uboLayoutBinding.stageFlags = VK_SHADER_STAGE_VERTEX_BIT;

		VkDescriptorSetLayoutCreateInfo layoutInfo = {};
		layoutInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
		layoutInfo.bindingCount = 1;
		layoutInfo.pBindings = &uboLayoutBinding;
		handle = device_->createDescriptorSetLayout(&layoutInfo);
	}
};

//only used by VulkanGraphicsPipeline's ctor
struct VulkanShaderModule : public VulkanHandle<VkShaderModule> {
protected:
	//held:
	VkDevice device = {};
public:
	~VulkanShaderModule() {
		if (handle) vkDestroyShaderModule(device, handle, getAllocator());
	}
	
	VulkanShaderModule(
		VulkanLogicalDevice const * const device_,
		std::string const code
	) : device((*device_)()) {
		VkShaderModuleCreateInfo createInfo = {};
		createInfo.sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO;
		createInfo.codeSize = code.length();
		createInfo.pCode = reinterpret_cast<const uint32_t*>(code.data());
		handle = device_->createShaderModule(&createInfo);
	}
};

struct VulkanGraphicsPipeline : public VulkanHandle<VkPipeline> {
protected:
	//owned:
	VkPipelineLayout pipelineLayout = {};
	std::unique_ptr<VulkanDescriptorSetLayout> descriptorSetLayout;
	
	//held:
	VkDevice device = {};				//held for dtor
public:
	VkPipelineLayout getPipelineLayout() const { return pipelineLayout; }
	
	VulkanDescriptorSetLayout * getDescriptorSetLayout() { return descriptorSetLayout.get(); }
	VulkanDescriptorSetLayout const * getDescriptorSetLayout() const { return descriptorSetLayout.get(); }

	~VulkanGraphicsPipeline() {
		if (pipelineLayout) vkDestroyPipelineLayout(device, pipelineLayout, getAllocator());
		if (handle) vkDestroyPipeline(device, handle, getAllocator());
		descriptorSetLayout = nullptr;
	}

	VulkanGraphicsPipeline(
		VulkanLogicalDevice const * const device_,
		VkRenderPass renderPass
	) : device((*device_)()) {
		auto vertShaderModule = VulkanShaderModule(
			device_,
			Common::File::read("shader-vert.spv")
		);
		VkPipelineShaderStageCreateInfo vertShaderStageInfo = {};
		vertShaderStageInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
		vertShaderStageInfo.stage = VK_SHADER_STAGE_VERTEX_BIT;
		vertShaderStageInfo.module = vertShaderModule();
		vertShaderStageInfo.pName = "main";
		// GLSL uses 'main', but clspv doesn't allow 'main', so ....
		//vertShaderStageInfo.pName = "vert";
		
		auto fragShaderModule = VulkanShaderModule(
			device_,
			Common::File::read("shader-frag.spv")
		);
		VkPipelineShaderStageCreateInfo fragShaderStageInfo = {};
		fragShaderStageInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
		fragShaderStageInfo.stage = VK_SHADER_STAGE_FRAGMENT_BIT;
		fragShaderStageInfo.module = fragShaderModule();
		fragShaderStageInfo.pName = "main";
		//fragShaderStageInfo.pName = "frag";

		auto bindingDescription = Vertex::getBindingDescription();
		auto attributeDescriptions = Vertex::getAttributeDescriptions();

		VkPipelineVertexInputStateCreateInfo vertexInputInfo = {};
		vertexInputInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO;
		vertexInputInfo.vertexBindingDescriptionCount = 1;
		vertexInputInfo.pVertexBindingDescriptions = &bindingDescription;
		vertexInputInfo.vertexAttributeDescriptionCount = static_cast<uint32_t>(attributeDescriptions.size());
		vertexInputInfo.pVertexAttributeDescriptions = attributeDescriptions.data();

		VkPipelineInputAssemblyStateCreateInfo inputAssembly = {};
		inputAssembly.sType = VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO;
		inputAssembly.topology = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST;
		inputAssembly.primitiveRestartEnable = VK_FALSE;

		VkPipelineViewportStateCreateInfo viewportState = {};
		viewportState.sType = VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO;
		viewportState.viewportCount = 1;
		viewportState.scissorCount = 1;

		VkPipelineRasterizationStateCreateInfo rasterizer = {};
		rasterizer.sType = VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO;
		rasterizer.depthClampEnable = VK_FALSE;
		rasterizer.rasterizerDiscardEnable = VK_FALSE;
		rasterizer.polygonMode = VK_POLYGON_MODE_FILL;
		rasterizer.lineWidth = 1;
		rasterizer.cullMode = VK_CULL_MODE_BACK_BIT;
		//rasterizer.frontFace = VK_FRONT_FACE_CLOCKWISE;
		// this was changed in lesson 23 ...
		rasterizer.frontFace = VK_FRONT_FACE_COUNTER_CLOCKWISE;
		rasterizer.depthBiasEnable = VK_FALSE;

		VkPipelineMultisampleStateCreateInfo multisampling = {};
		multisampling.sType = VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO;
		multisampling.sampleShadingEnable = VK_FALSE;
		multisampling.rasterizationSamples = VK_SAMPLE_COUNT_1_BIT;

		VkPipelineColorBlendAttachmentState colorBlendAttachment = {};
		colorBlendAttachment.colorWriteMask = VK_COLOR_COMPONENT_R_BIT | VK_COLOR_COMPONENT_G_BIT | VK_COLOR_COMPONENT_B_BIT | VK_COLOR_COMPONENT_A_BIT;
		colorBlendAttachment.blendEnable = VK_FALSE;

		VkPipelineColorBlendStateCreateInfo colorBlending = {};
		colorBlending.sType = VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO;
		colorBlending.logicOpEnable = VK_FALSE;
		colorBlending.logicOp = VK_LOGIC_OP_COPY;
		colorBlending.attachmentCount = 1;
		colorBlending.pAttachments = &colorBlendAttachment;
		colorBlending.blendConstants[0] = 0;
		colorBlending.blendConstants[1] = 0;
		colorBlending.blendConstants[2] = 0;
		colorBlending.blendConstants[3] = 0;

		std::vector<VkDynamicState> dynamicStates = {
			VK_DYNAMIC_STATE_VIEWPORT,
			VK_DYNAMIC_STATE_SCISSOR
		};
		VkPipelineDynamicStateCreateInfo dynamicState = {};
		dynamicState.sType = VK_STRUCTURE_TYPE_PIPELINE_DYNAMIC_STATE_CREATE_INFO;
		dynamicState.dynamicStateCount = static_cast<uint32_t>(dynamicStates.size());
		dynamicState.pDynamicStates = dynamicStates.data();
		
		// descriptorSetLayout is only used by graphicsPipeline
		VkPipelineLayoutCreateInfo pipelineLayoutInfo = {};
		pipelineLayoutInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
#if 1
		descriptorSetLayout = std::make_unique<VulkanDescriptorSetLayout>(device_);
		std::vector<VkDescriptorSetLayout> descriptorSetLayouts = {
			(*descriptorSetLayout)(),
		};
		pipelineLayoutInfo.setLayoutCount = static_cast<uint32_t>(descriptorSetLayouts.size());
		pipelineLayoutInfo.pSetLayouts = descriptorSetLayouts.data();
#else
		pipelineLayoutInfo.setLayoutCount = 1;
		pipelineLayoutInfo.pSetLayouts = &(*descriptorSetLayout)();
#endif
		pipelineLayout = device_->createPipelineLayout(&pipelineLayoutInfo);

		VkPipelineShaderStageCreateInfo shaderStages[] = {
			vertShaderStageInfo,
			fragShaderStageInfo,
		};
	
		VkGraphicsPipelineCreateInfo pipelineInfo = {};
		pipelineInfo.sType = VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO;
		pipelineInfo.stageCount = numberof(shaderStages);
		pipelineInfo.pStages = shaderStages;
		pipelineInfo.pVertexInputState = &vertexInputInfo;
		pipelineInfo.pInputAssemblyState = &inputAssembly;
		pipelineInfo.pViewportState = &viewportState;
		pipelineInfo.pRasterizationState = &rasterizer;
		pipelineInfo.pMultisampleState = &multisampling;
		pipelineInfo.pColorBlendState = &colorBlending;
		pipelineInfo.pDynamicState = &dynamicState;
		pipelineInfo.layout = pipelineLayout;
		pipelineInfo.renderPass = renderPass;
		pipelineInfo.subpass = 0;
		pipelineInfo.basePipelineHandle = VK_NULL_HANDLE;
		handle = device_->createGraphicsPipelines((VkPipelineCache)VK_NULL_HANDLE, 1, &pipelineInfo, nullptr);
	}
};

struct VulkanCommandPool : public VulkanHandle<VkCommandPool> {
protected:
	//held:
	VkDevice device = {};
public:
	~VulkanCommandPool() {
		if (handle) vkDestroyCommandPool(device, handle, getAllocator());
	}
	VulkanCommandPool(
		VulkanPhysicalDevice const * const physicalDevice,
		VkSurfaceKHR surface,
		VulkanLogicalDevice const * const device_
	) : device((*device_)()) {
		auto queueFamilyIndices = physicalDevice->findQueueFamilies(surface);
		VkCommandPoolCreateInfo poolInfo = {};
		poolInfo.sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO;
		poolInfo.flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;
		poolInfo.queueFamilyIndex = queueFamilyIndices.graphicsFamily.value();
		handle = device_->createCommandPool(&poolInfo);
	}
};

// methods common to VkBuffer and VkImage
template<typename Handle, auto Destroy>
struct VulkanDeviceMemoryParent : public VulkanHandle<Handle> {
	using Super = VulkanHandle<Handle>;
protected:
	//owns
	VkDeviceMemory memory = {};
	//holds
	VulkanLogicalDevice const * device = {};
public:
	auto const & getMemory() const { return memory; }
	auto & getMemory() { return memory; }

	~VulkanDeviceMemoryParent() {
		if (memory) vkFreeMemory((*device)(), memory, nullptr);
		if (Super::handle) Destroy((*device)(), Super::handle, nullptr);
	}

	VulkanDeviceMemoryParent(
		Handle handle_,
		VkDeviceMemory memory_,
		VulkanLogicalDevice const * const device_
	) : VulkanHandle<Handle>(handle_),
		memory(memory_),
		device(device_)
	{}

	//only call this after handle is filled
	// is this a 'device' method or a 'buffer' method?
	VkMemoryRequirements getMemoryRequirements() const {
		VkMemoryRequirements memRequirements = {};
		vkGetBufferMemoryRequirements((*device)(), Super::handle, &memRequirements);
		return memRequirements;
	}
};

struct VulkanDeviceMemoryBuffer : public VulkanDeviceMemoryParent<VkBuffer, vkDestroyBuffer> {
	using Super = VulkanDeviceMemoryParent<VkBuffer, vkDestroyBuffer>;
	
	using Super::Super;

	//ctor for VkBuffer's whether they are being ctor'd by staging or by uniforms whatever
	VulkanDeviceMemoryBuffer(
		VulkanPhysicalDevice const * const physicalDevice,
		VulkanLogicalDevice const * const device_,
		VkDeviceSize size,
		VkBufferUsageFlags usage,
		VkMemoryPropertyFlags properties
	) : Super({}, {}, device_) {
		
		VkBufferCreateInfo bufferInfo = {};
		bufferInfo.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
		bufferInfo.size = size;
		bufferInfo.usage = usage;
		bufferInfo.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
		VulkanHandle<Handle>::handle = device_->createBuffer(&bufferInfo);

		VkMemoryRequirements memRequirements = Super::getMemoryRequirements();
		VkMemoryAllocateInfo allocInfo = {};
		allocInfo.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
		allocInfo.allocationSize = memRequirements.size;
		allocInfo.memoryTypeIndex = physicalDevice->findMemoryType(memRequirements.memoryTypeBits, properties);
		VULKAN_SAFE(vkAllocateMemory, (*Super::device)(), &allocInfo, nullptr, &Super::getMemory());

		vkBindBufferMemory((*Super::device)(), (*this)(), Super::memory, 0);
	}

	// TODO make a StagingBuffer subclass? 
	static std::unique_ptr<VulkanDeviceMemoryBuffer> makeFromStaged(
		VulkanPhysicalDevice const * const physicalDevice,
		VulkanLogicalDevice const * const device,
		void const * const srcData,
		size_t bufferSize
	) {
		auto stagingBuffer = std::make_unique<VulkanDeviceMemoryBuffer>(
			physicalDevice,
			device,
			bufferSize,
			VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
			VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT
		);

		void * dstData = {};
		vkMapMemory(
			(*device)(),
			stagingBuffer->getMemory(),
			0,
			bufferSize,
			0,
			&dstData
		);
		memcpy(dstData, srcData, (size_t)bufferSize);
		vkUnmapMemory((*device)(), stagingBuffer->getMemory());

		return stagingBuffer;
	}

public:
	// requires Handle == VkBuffer
	static std::unique_ptr<VulkanDeviceMemoryBuffer> makeBufferFromStaged(
		VulkanPhysicalDevice const * const physicalDevice,
		VulkanLogicalDevice const * const device,
		VulkanCommandPool const * const commandPool,
		void const * const srcData,
		size_t bufferSize
	) {
		std::unique_ptr<VulkanDeviceMemoryBuffer> stagingBuffer = makeFromStaged(
			physicalDevice,
			device,
			srcData,
			bufferSize
		);

		auto buffer = std::make_unique<VulkanDeviceMemoryBuffer>(
			physicalDevice,
			device,
			bufferSize,
			VK_BUFFER_USAGE_TRANSFER_DST_BIT | VK_BUFFER_USAGE_VERTEX_BUFFER_BIT,
			VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT
		);
		
		copyBuffer(
			device,
			commandPool,
			(*stagingBuffer)(),
			(*buffer)(),
			bufferSize
		);
	
		return buffer;
	}

protected:
	//copies based on the graphicsQueue
	// used by makeBufferFromStaged
	static void copyBuffer(
		VulkanLogicalDevice const * const device,
		VulkanCommandPool const * const commandPool,
		VkBuffer srcBuffer,	//staging VkBuffer
		Handle dstBuffer,	//dest VkBuffer
		VkDeviceSize size
	) {
		VkCommandBufferAllocateInfo allocInfo = {};
		allocInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
		allocInfo.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
		allocInfo.commandPool = (*commandPool)();
		allocInfo.commandBufferCount = 1;

		VkCommandBuffer commandBuffer;
		vkAllocateCommandBuffers((*device)(), &allocInfo, &commandBuffer);

		VkCommandBufferBeginInfo beginInfo = {};
		beginInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
		beginInfo.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;

		vkBeginCommandBuffer(commandBuffer, &beginInfo);

		{
			VkBufferCopy copyRegion = {};
			copyRegion.size = size;
			vkCmdCopyBuffer(commandBuffer, srcBuffer, dstBuffer, 1, &copyRegion);
		}

		vkEndCommandBuffer(commandBuffer);

		VkSubmitInfo submitInfo = {};
		submitInfo.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
		submitInfo.commandBufferCount = 1;
		submitInfo.pCommandBuffers = &commandBuffer;

		vkQueueSubmit(device->graphicsQueue, 1, &submitInfo, VK_NULL_HANDLE);
		vkQueueWaitIdle(device->graphicsQueue);

		vkFreeCommandBuffers((*device)(), (*commandPool)(), 1, &commandBuffer);
	}
};


struct VulkanDeviceMemoryImage : public VulkanDeviceMemoryParent<VkImage, vkDestroyImage> {
	using Super = VulkanDeviceMemoryParent<VkImage, vkDestroyImage>;
	
	using Super::Super;

public:
	// requires Handle == VkImage
	static std::unique_ptr<VulkanDeviceMemoryImage>
	makeTextureFromStaged(
		VulkanPhysicalDevice const * const physicalDevice,
		VulkanLogicalDevice const * const device,
		VulkanCommandPool const * const commandPool,
		void const * const srcData,
		size_t bufferSize,
		int texWidth,
		int texHeight
	) {
		std::unique_ptr<VulkanDeviceMemoryBuffer> stagingBuffer 
		= VulkanDeviceMemoryBuffer::makeFromStaged(
			physicalDevice,
			device,
			srcData,
			bufferSize
		);
		
		// but here we have the destination isn't a VkBuffer, it's a VkImage
		auto image = createImage(
			physicalDevice,
			device,
			texWidth,
			texHeight,
			VK_FORMAT_R8G8B8A8_SRGB,
			VK_IMAGE_TILING_OPTIMAL,
			VK_IMAGE_USAGE_TRANSFER_DST_BIT | VK_IMAGE_USAGE_SAMPLED_BIT,
			VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT
		);

		image->transitionImageLayout(
			commandPool,
			VK_FORMAT_R8G8B8A8_SRGB,
			VK_IMAGE_LAYOUT_UNDEFINED,
			VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL
		);
		copyBufferToImage(
			device,
			commandPool,
			(*stagingBuffer)(),
			(*image)(),
			static_cast<uint32_t>(texWidth),
			static_cast<uint32_t>(texHeight)
		);
		image->transitionImageLayout(
			commandPool,
			VK_FORMAT_R8G8B8A8_SRGB,
			VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
			VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL
		);
	
		return image;
	}

protected:
	static std::unique_ptr<VulkanDeviceMemoryImage> createImage(
		VulkanPhysicalDevice const * const physicalDevice,
		VulkanLogicalDevice const * const device,
		uint32_t width,
		uint32_t height,
		VkFormat format,
		VkImageTiling tiling,
		VkImageUsageFlags usage,
		VkMemoryPropertyFlags properties
	) {
		VkImage image = {};
		VkDeviceMemory imageMemory = {};

		VkImageCreateInfo imageInfo = {};
		imageInfo.sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO;
		imageInfo.imageType = VK_IMAGE_TYPE_2D;
		imageInfo.extent.width = width;
		imageInfo.extent.height = height;
		imageInfo.extent.depth = 1;
		imageInfo.mipLevels = 1;
		imageInfo.arrayLayers = 1;
		imageInfo.format = format;
		imageInfo.tiling = tiling;
		imageInfo.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
		imageInfo.usage = usage;
		imageInfo.samples = VK_SAMPLE_COUNT_1_BIT;
		imageInfo.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
		VULKAN_SAFE(vkCreateImage, (*device)(), &imageInfo, nullptr, &image);

		VkMemoryRequirements memRequirements;
		vkGetImageMemoryRequirements((*device)(), image, &memRequirements);

		VkMemoryAllocateInfo allocInfo{};
		allocInfo.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
		allocInfo.allocationSize = memRequirements.size;
		allocInfo.memoryTypeIndex = physicalDevice->findMemoryType(memRequirements.memoryTypeBits, properties);

		VULKAN_SAFE(vkAllocateMemory, (*device)(), &allocInfo, nullptr, &imageMemory);

		vkBindImageMemory((*device)(), image, imageMemory, 0);
		
		return std::make_unique<VulkanDeviceMemoryImage>(image, imageMemory, device);
	}

	void transitionImageLayout(
		VulkanCommandPool const * const commandPool,
		VkFormat format,
		VkImageLayout oldLayout,
		VkImageLayout newLayout
	) {
		VkCommandBuffer commandBuffer = beginSingleTimeCommands(Super::device, commandPool);

		VkImageMemoryBarrier barrier{};
		barrier.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
		barrier.oldLayout = oldLayout;
		barrier.newLayout = newLayout;
		barrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
		barrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
		barrier.image = (*this)();
		barrier.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
		barrier.subresourceRange.baseMipLevel = 0;
		barrier.subresourceRange.levelCount = 1;
		barrier.subresourceRange.baseArrayLayer = 0;
		barrier.subresourceRange.layerCount = 1;

		VkPipelineStageFlags sourceStage;
		VkPipelineStageFlags destinationStage;

		if (oldLayout == VK_IMAGE_LAYOUT_UNDEFINED && newLayout == VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL) {
			barrier.srcAccessMask = 0;
			barrier.dstAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;

			sourceStage = VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT;
			destinationStage = VK_PIPELINE_STAGE_TRANSFER_BIT;
		} else if (oldLayout == VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL && newLayout == VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL) {
			barrier.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
			barrier.dstAccessMask = VK_ACCESS_SHADER_READ_BIT;

			sourceStage = VK_PIPELINE_STAGE_TRANSFER_BIT;
			destinationStage = VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT;
		} else {
			throw std::invalid_argument("unsupported layout transition!");
		}

		vkCmdPipelineBarrier(
			commandBuffer,
			sourceStage, destinationStage,
			0,
			0, nullptr,
			0, nullptr,
			1, &barrier
		);

		endSingleTimeCommands(Super::device, commandPool, commandBuffer);
	}

	static void copyBufferToImage(
		VulkanLogicalDevice const * const device,
		VulkanCommandPool const * const commandPool,
		VkBuffer buffer,
		VkImage image,
		uint32_t width,
		uint32_t height
	) {
		VkCommandBuffer commandBuffer = beginSingleTimeCommands(device, commandPool);

		VkBufferImageCopy region{};
		region.bufferOffset = 0;
		region.bufferRowLength = 0;
		region.bufferImageHeight = 0;
		region.imageSubresource.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
		region.imageSubresource.mipLevel = 0;
		region.imageSubresource.baseArrayLayer = 0;
		region.imageSubresource.layerCount = 1;
		region.imageOffset = {0, 0, 0};
		region.imageExtent = {
			width,
			height,
			1
		};

		vkCmdCopyBufferToImage(commandBuffer, buffer, image, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 1, &region);

		endSingleTimeCommands(device, commandPool, commandBuffer);
	}

	static VkCommandBuffer beginSingleTimeCommands(
		VulkanLogicalDevice const * const device,
		VulkanCommandPool const * const commandPool
	) {
		VkCommandBufferAllocateInfo allocInfo{};
		allocInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
		allocInfo.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
		allocInfo.commandPool = (*commandPool)();
		allocInfo.commandBufferCount = 1;

		VkCommandBuffer commandBuffer;
		vkAllocateCommandBuffers((*device)(), &allocInfo, &commandBuffer);

		VkCommandBufferBeginInfo beginInfo{};
		beginInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
		beginInfo.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;

		vkBeginCommandBuffer(commandBuffer, &beginInfo);

		return commandBuffer;
	}

	static void endSingleTimeCommands(
		VulkanLogicalDevice const * const device,
		VulkanCommandPool const * const commandPool,
		VkCommandBuffer commandBuffer
	) {
		vkEndCommandBuffer(commandBuffer);

		VkSubmitInfo submitInfo{};
		submitInfo.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
		submitInfo.commandBufferCount = 1;
		submitInfo.pCommandBuffers = &commandBuffer;

		vkQueueSubmit(device->graphicsQueue, 1, &submitInfo, VK_NULL_HANDLE);
		vkQueueWaitIdle(device->graphicsQueue);

		vkFreeCommandBuffers((*device)(), (*commandPool)(), 1, &commandBuffer);
	}
};

// so I don't have to prefix all my fields and names
struct VulkanCommon {
	::SDLApp::SDLApp const * app = {};	// points back to the owner

#if 0	// not working on my vulkan implementation
	static constexpr bool const enableValidationLayers = true;
#else
	static constexpr bool const enableValidationLayers = false;
#endif

	std::unique_ptr<VulkanInstance> instance;
	std::unique_ptr<VulkanDebugMessenger> debug;	// optional
	std::unique_ptr<VulkanSurface> surface;
	std::unique_ptr<VulkanLogicalDevice> device;
	std::unique_ptr<VulkanSwapChain> swapChain;
	std::unique_ptr<VulkanGraphicsPipeline> graphicsPipeline;
	std::unique_ptr<VulkanCommandPool> commandPool;
	std::unique_ptr<VulkanDeviceMemoryBuffer> vertexBuffer;
	std::unique_ptr<VulkanDeviceMemoryBuffer> indexBuffer;
	std::unique_ptr<VulkanDeviceMemoryImage> textureImage;

	// hmm should the map be a field?
	std::vector<std::unique_ptr<VulkanDeviceMemoryBuffer>> uniformBuffers;
	std::vector<void*> uniformBuffersMapped;

	VkDescriptorPool descriptorPool = {};
	std::vector<VkDescriptorSet> descriptorSets;

	std::vector<VkCommandBuffer> commandBuffers;
	std::vector<VkSemaphore> imageAvailableSemaphores;
	std::vector<VkSemaphore> renderFinishedSemaphores;
	std::vector<VkFence> inFlightFences;
	uint32_t currentFrame = {};
	
	bool framebufferResized = {};

	// used by
	//	VulkanPhysicalDevice::checkDeviceExtensionSupport
	//	initLogicalDevice
	std::vector<char const *> const deviceExtensions = {
		VK_KHR_SWAPCHAIN_EXTENSION_NAME
	};

	//ok now we're at the point where we are recreating objects dependent on physicalDevice so
	std::unique_ptr<VulkanPhysicalDevice> physicalDevice;

	VulkanCommon(::SDLApp::SDLApp const * const app_)
	: app(app_) {
		if (enableValidationLayers && !checkValidationLayerSupport()) {
			throw Common::Exception() << "validation layers requested, but not available!";
		}

		// hmm, maybe instance should be a shared_ptr and then passed to debug, surface, and physicalDevice ?
		instance = std::make_unique<VulkanInstance>(app, enableValidationLayers);
		
		if (enableValidationLayers) {
			debug = std::make_unique<VulkanDebugMessenger>(instance.get());
		}
		
		surface = std::make_unique<VulkanSurface>(app->getWindow(), (*instance)());
		
		physicalDevice = std::make_unique<VulkanPhysicalDevice>(instance.get(), (*surface)(), deviceExtensions);
		device = std::make_unique<VulkanLogicalDevice>(physicalDevice.get(), (*surface)(), deviceExtensions, enableValidationLayers);
		
		swapChain = std::make_unique<VulkanSwapChain>(
			app->getScreenSize(),
			physicalDevice.get(),
			device.get(),
			(*surface)()
		);
		
		graphicsPipeline = std::make_unique<VulkanGraphicsPipeline>(
			device.get(),
			swapChain->getRenderPass()
		);
		
		commandPool = std::make_unique<VulkanCommandPool>(
			physicalDevice.get(),
			(*surface)(),
			device.get()
		);
		
		createTextureImage();
		
		initVertexBuffer();
		initIndexBuffer();
		initUniformBuffers();
		createDescriptorPool();
		createDescriptorSets();
		initCommandBuffers();
		initSyncObjects();
	}

protected:
	// this is out of place
	static bool checkValidationLayerSupport() {
		auto availableLayers = vulkanEnum<VkLayerProperties>(
			NAME_PAIR(vkEnumerateInstanceLayerProperties)
		);
		for (char const * const layerName : validationLayers) {
			bool layerFound = false;
			for (auto const & layerProperties : availableLayers) {
				if (!strcmp(layerName, layerProperties.layerName)) {
					layerFound = true;
					break;
				}
			}
			if (!layerFound) {
				return false;
			}
		}
		return true;
	}

	static constexpr int MAX_FRAMES_IN_FLIGHT = 2;

public:
	~VulkanCommon() {
		swapChain = nullptr;
		graphicsPipeline = nullptr;

		uniformBuffers.clear();
		indexBuffer = nullptr;
		vertexBuffer = nullptr;

		vkDestroyDescriptorPool((*device)(), descriptorPool, nullptr);
	
		textureImage = nullptr;

		for (size_t i = 0; i < MAX_FRAMES_IN_FLIGHT; i++) {
			vkDestroySemaphore((*device)(), renderFinishedSemaphores[i], nullptr);
			vkDestroySemaphore((*device)(), imageAvailableSemaphores[i], nullptr);
			vkDestroyFence((*device)(), inFlightFences[i], nullptr);
		}

		commandPool = nullptr;
		device = nullptr;
		surface = nullptr;
		debug = nullptr;
		instance = nullptr;
	}

protected:
	void createTextureImage() {
		std::string filename = "texture.jpg";
		std::shared_ptr<Image::Image> image = std::dynamic_pointer_cast<Image::Image>(Image::system->read(filename));
		if (!image) {
			throw Common::Exception() << "failed to load image from " << filename;
		}
		int texWidth = image->getSize().x;
		int texHeight = image->getSize().y;
		int texBPP = image->getBitsPerPixel() >> 3;
		int desiredBPP = 4;
		if (texBPP != desiredBPP) {
			//resample
			auto newimage = std::make_shared<Image::Image>(image->getSize(), nullptr, desiredBPP);
			for (int i = 0; i < texWidth * texHeight; ++i) {
				int j = 0;
				for (; j < texBPP && j < desiredBPP; ++j) {
					newimage->getData()[desiredBPP*i+j] = image->getData()[texBPP*i+j];
				}
				for (; j < desiredBPP; ++j) {
					newimage->getData()[desiredBPP*i+j] = 255;
				}
			}
			image = newimage;
			texBPP = image->getBitsPerPixel() >> 3;
		}
		char * srcData = image->getData();
		VkDeviceSize bufferSize = texWidth * texHeight * texBPP;
		
		textureImage = VulkanDeviceMemoryImage::makeTextureFromStaged(
			physicalDevice.get(),
			device.get(),
			commandPool.get(),
			srcData,
			bufferSize,
			texWidth,
			texHeight
		);
	}


	void recreateSwapChain() {
		
#if 0 //hmm why are there multiple events?
		int width = app->getScreenSize().x;
		int height = app->getScreenSize().y;
		glfwGetFramebufferSize(window, &width, &height);
		while (width == 0 || height == 0) {
			glfwGetFramebufferSize(window, &width, &height);
			glfwWaitEvents();
		}
#else
		if (!app->getScreenSize().x ||
			!app->getScreenSize().y)
		{
			throw Common::Exception() << "here";
		}
#endif
		device->waitIdle();

		swapChain = std::make_unique<VulkanSwapChain>(
			app->getScreenSize(),
			physicalDevice.get(),
			device.get(),
			(*surface)()
		);
	}

	void initVertexBuffer() {
		vertexBuffer = VulkanDeviceMemoryBuffer::makeBufferFromStaged(
			physicalDevice.get(),
			device.get(),
			commandPool.get(),
			vertices.data(),
			sizeof(vertices[0]) * vertices.size()
		);
	}

	void initIndexBuffer() {
		indexBuffer = VulkanDeviceMemoryBuffer::makeBufferFromStaged(
			physicalDevice.get(),
			device.get(),
			commandPool.get(),
			indices.data(),
			sizeof(indices[0]) * indices.size()
		);
	}

	void initUniformBuffers() {
		VkDeviceSize bufferSize = sizeof(UniformBufferObject);

		uniformBuffers.resize(MAX_FRAMES_IN_FLIGHT);
		uniformBuffersMapped.resize(MAX_FRAMES_IN_FLIGHT);

		for (size_t i = 0; i < MAX_FRAMES_IN_FLIGHT; i++) {
			uniformBuffers[i] = std::make_unique<VulkanDeviceMemoryBuffer>(
				physicalDevice.get(),
				device.get(),
				bufferSize,
				VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT,
				VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT
			);
			
			vkMapMemory(
				(*device)(),
				uniformBuffers[i]->getMemory(),
				0,
				bufferSize,
				0,
				&uniformBuffersMapped[i]
			);
		}
	}

	void initCommandBuffers() {
		commandBuffers.resize(MAX_FRAMES_IN_FLIGHT);
		VkCommandBufferAllocateInfo allocInfo = {};
		allocInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
		allocInfo.commandPool = (*commandPool)();
		allocInfo.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
		allocInfo.commandBufferCount = (uint32_t)commandBuffers.size();
		VULKAN_SAFE(vkAllocateCommandBuffers, (*device)(), &allocInfo, commandBuffers.data());
	}

	void recordCommandBuffer(VkCommandBuffer commandBuffer, uint32_t imageIndex) {
		VkCommandBufferBeginInfo beginInfo = {};
		beginInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;

		VULKAN_SAFE(vkBeginCommandBuffer, commandBuffer, &beginInfo);

		VkRenderPassBeginInfo renderPassInfo = {};
		renderPassInfo.sType = VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO;
		renderPassInfo.renderPass = swapChain->getRenderPass();
		renderPassInfo.framebuffer = swapChain->framebuffers[imageIndex];
		renderPassInfo.renderArea.offset = {0, 0};
		renderPassInfo.renderArea.extent = swapChain->extent;

		VkClearValue clearColor = {{{0, 0, 0, 1}}};
		renderPassInfo.clearValueCount = 1;
		renderPassInfo.pClearValues = &clearColor;

		vkCmdBeginRenderPass(commandBuffer, &renderPassInfo, VK_SUBPASS_CONTENTS_INLINE);

		{
			vkCmdBindPipeline(commandBuffer, VK_PIPELINE_BIND_POINT_GRAPHICS, (*graphicsPipeline)());

			VkViewport viewport = {};
			viewport.x = 0;
			viewport.y = 0;
			viewport.width = (float)swapChain->extent.width;
			viewport.height = (float)swapChain->extent.height;
			viewport.minDepth = 0;
			viewport.maxDepth = 1;
			vkCmdSetViewport(commandBuffer, 0, 1, &viewport);

			VkRect2D scissor = {};
			scissor.offset = {0, 0};
			scissor.extent = swapChain->extent;
			vkCmdSetScissor(commandBuffer, 0, 1, &scissor);

			VkBuffer vertexBuffers[] = {(*vertexBuffer)()};
			VkDeviceSize offsets[] = {0};
			vkCmdBindVertexBuffers(commandBuffer, 0, numberof(vertexBuffers), vertexBuffers, offsets);

			vkCmdBindIndexBuffer(commandBuffer, (*indexBuffer)(), 0, VK_INDEX_TYPE_UINT16);

			vkCmdBindDescriptorSets(
				commandBuffer,
				VK_PIPELINE_BIND_POINT_GRAPHICS,
				graphicsPipeline->getPipelineLayout(),
				0,
				1,
				&descriptorSets[currentFrame],
				0,
				nullptr
			);

// hmm when I use uniforms it is crashing here ...
			vkCmdDrawIndexed(commandBuffer, static_cast<uint32_t>(indices.size()), 1, 0, 0, 0);
		}

		vkCmdEndRenderPass(commandBuffer);

		VULKAN_SAFE(vkEndCommandBuffer, commandBuffer);
	}

	void initSyncObjects() {
		imageAvailableSemaphores.resize(MAX_FRAMES_IN_FLIGHT);
		renderFinishedSemaphores.resize(MAX_FRAMES_IN_FLIGHT);
		inFlightFences.resize(MAX_FRAMES_IN_FLIGHT);

		VkSemaphoreCreateInfo semaphoreInfo = {};
		semaphoreInfo.sType = VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO;

		VkFenceCreateInfo fenceInfo = {};
		fenceInfo.sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO;
		fenceInfo.flags = VK_FENCE_CREATE_SIGNALED_BIT;

		for (size_t i = 0; i < MAX_FRAMES_IN_FLIGHT; i++) {
			VULKAN_SAFE(vkCreateSemaphore, (*device)(), &semaphoreInfo, nullptr, &imageAvailableSemaphores[i]);
			VULKAN_SAFE(vkCreateSemaphore, (*device)(), &semaphoreInfo, nullptr, &renderFinishedSemaphores[i]);
			VULKAN_SAFE(vkCreateFence, (*device)(), &fenceInfo, nullptr, &inFlightFences[i]);
		}
	}

	void createDescriptorPool() {
		VkDescriptorPoolSize poolSize{};
		poolSize.type = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
		poolSize.descriptorCount = static_cast<uint32_t>(MAX_FRAMES_IN_FLIGHT);

		VkDescriptorPoolCreateInfo poolInfo{};
		poolInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
		poolInfo.poolSizeCount = 1;
		poolInfo.pPoolSizes = &poolSize;
		poolInfo.maxSets = static_cast<uint32_t>(MAX_FRAMES_IN_FLIGHT);
		VULKAN_SAFE(vkCreateDescriptorPool, (*device)(), &poolInfo, nullptr, &descriptorPool);
	}

	void createDescriptorSets() {
		std::vector<VkDescriptorSetLayout> layouts(MAX_FRAMES_IN_FLIGHT, (*graphicsPipeline->getDescriptorSetLayout())());
		VkDescriptorSetAllocateInfo allocInfo{};
		allocInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
		allocInfo.descriptorPool = descriptorPool;
		allocInfo.descriptorSetCount = static_cast<uint32_t>(MAX_FRAMES_IN_FLIGHT);
		allocInfo.pSetLayouts = layouts.data();
		descriptorSets.resize(MAX_FRAMES_IN_FLIGHT);
		VULKAN_SAFE(vkAllocateDescriptorSets, (*device)(), &allocInfo, descriptorSets.data());

		for (size_t i = 0; i < MAX_FRAMES_IN_FLIGHT; i++) {
			VkDescriptorBufferInfo bufferInfo{};
			bufferInfo.buffer = (*uniformBuffers[i])();
			bufferInfo.offset = 0;
			bufferInfo.range = sizeof(UniformBufferObject);

			VkWriteDescriptorSet descriptorWrite{};
			descriptorWrite.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
			descriptorWrite.dstSet = descriptorSets[i];
			descriptorWrite.dstBinding = 0;
			descriptorWrite.dstArrayElement = 0;
			descriptorWrite.descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
			descriptorWrite.descriptorCount = 1;
			descriptorWrite.pBufferInfo = &bufferInfo;
			vkUpdateDescriptorSets((*device)(), 1, &descriptorWrite, 0, nullptr);
		}
	}


	//decltype(std::chrono::high_resolution_clock::now()) startTime = std::chrono::high_resolution_clock::now();
	
	void updateUniformBuffer(uint32_t currentFrame_) {
		static auto startTime = std::chrono::high_resolution_clock::now();
		auto currentTime = std::chrono::high_resolution_clock::now();
		float time = std::chrono::duration<float, std::chrono::seconds::period>(currentTime - startTime).count();

		UniformBufferObject ubo = {};
		ubo.model = Tensor::rotate<float>(
			Tensor::_ident<float,4>(),
			time * degToRad<float>(90),
			Tensor::float3(0, 0, 1)
		);
		ubo.view = Tensor::lookAt<float>(
			Tensor::float3(2, 2, 2),
			Tensor::float3(0, 0, 0),
			Tensor::float3(0, 0, 1)
		);
		ubo.proj = Tensor::perspective<float>(
			degToRad<float>(45),
			(float)swapChain->extent.width / (float)swapChain->extent.height,
			0.1f,
			10
		);
		ubo.proj[1][1] *= -1;

		memcpy(uniformBuffersMapped[currentFrame_], &ubo, sizeof(ubo));
	}
public:
	void drawFrame() {
		vkWaitForFences((*device)(), 1, &inFlightFences[currentFrame], VK_TRUE, UINT64_MAX);

		uint32_t imageIndex = {};
		VkResult result = vkAcquireNextImageKHR(
			(*device)(),
			(*swapChain)(),
			UINT64_MAX,
			imageAvailableSemaphores[currentFrame],
			VK_NULL_HANDLE,
			&imageIndex
		);
		if (result == VK_ERROR_OUT_OF_DATE_KHR) {
			recreateSwapChain();
			return;
		} else if (result != VK_SUCCESS && result != VK_SUBOPTIMAL_KHR) {
			throw Common::Exception() << "vkAcquireNextImageKHR failed: " << result;
		}
		
		updateUniformBuffer(currentFrame);

		vkResetFences((*device)(), 1, &inFlightFences[currentFrame]);

		vkResetCommandBuffer(commandBuffers[currentFrame], /*VkCommandBufferResetFlagBits*/ 0);
		recordCommandBuffer(commandBuffers[currentFrame], imageIndex);

		VkSubmitInfo submitInfo = {};
		submitInfo.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;

		VkSemaphore waitSemaphores[] = {imageAvailableSemaphores[currentFrame]};
		VkPipelineStageFlags waitStages[] = {VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT};
		submitInfo.waitSemaphoreCount = 1;
		submitInfo.pWaitSemaphores = waitSemaphores;
		submitInfo.pWaitDstStageMask = waitStages;

		submitInfo.commandBufferCount = 1;
		submitInfo.pCommandBuffers = &commandBuffers[currentFrame];

		VkSemaphore signalSemaphores[] = {renderFinishedSemaphores[currentFrame]};
		submitInfo.signalSemaphoreCount = 1;
		submitInfo.pSignalSemaphores = signalSemaphores;

		VULKAN_SAFE(vkQueueSubmit, device->graphicsQueue, 1, &submitInfo, inFlightFences[currentFrame]);

		VkPresentInfoKHR presentInfo = {};
		presentInfo.sType = VK_STRUCTURE_TYPE_PRESENT_INFO_KHR;

		presentInfo.waitSemaphoreCount = 1;
		presentInfo.pWaitSemaphores = signalSemaphores;

		VkSwapchainKHR swapChains[] = {(*swapChain)()};
		presentInfo.swapchainCount = 1;
		presentInfo.pSwapchains = swapChains;

		presentInfo.pImageIndices = &imageIndex;

		result = vkQueuePresentKHR(device->presentQueue, &presentInfo);
		if (result == VK_ERROR_OUT_OF_DATE_KHR || result == VK_SUBOPTIMAL_KHR || framebufferResized) {
			framebufferResized = false;
			recreateSwapChain();
		} else if (result != VK_SUCCESS) {
			throw Common::Exception() << "vkQueuePresentKHR failed: " << result;
		}

		currentFrame = (currentFrame + 1) % MAX_FRAMES_IN_FLIGHT;
	}
public:
	void loopDone() {
		vkDeviceWaitIdle((*device)());
	}
};

struct Test : public ::SDLApp::SDLApp {
	using Super = ::SDLApp::SDLApp;

protected:
	std::unique_ptr<VulkanCommon> vk;
	
	virtual void initWindow() {
		Super::initWindow();
		
		// TODO make the window not resizeable

		vk = std::make_unique<VulkanCommon>(this);
	}

	virtual std::string getTitle() const {
		return "Vulkan Test";
	}
	
	virtual Uint32 getSDLCreateWindowFlags() {
		auto flags = Super::getSDLCreateWindowFlags();
		flags |= SDL_WINDOW_VULKAN;
		flags &= ~SDL_WINDOW_RESIZABLE;
		return flags;
	}

	virtual void loop() {
		Super::loop();
		//why here instead of shutdown?

		vk->loopDone();
	}
	
	virtual void onUpdate() {
		Super::onUpdate();
		vk->drawFrame();
	}

	virtual void onResize() {
		vk->framebufferResized = true;
	}
};

SDLAPP_MAIN(Test)
