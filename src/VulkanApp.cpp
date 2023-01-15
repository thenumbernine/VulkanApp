#include "SDLApp/SDLApp.h"
#include "Image/Image.h"
#include "Common/Exception.h"
#include "Common/Macros.h"	//LINE_STRING
#include "Common/File.h"
#include "Common/Function.h"
#include "Tensor/Tensor.h"
#include <tiny_obj_loader.h>
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
		auto count = uint32_t{};
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
		auto count = uint32_t{};
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

auto assertHandle(auto x, char const * where) {
	if (!x) throw Common::Exception() << "returned an empty handle at " << where;
	return x;
}
#define ASSERTHANDLE(x) assertHandle(x, FILE_AND_LINE)

//TODO put this somewhere
namespace Common {

//https://stackoverflow.com/questions/26351587/how-to-create-stdarray-with-initialization-list-without-providing-size-directl
template <typename... T>
constexpr auto make_array(T&&... values)
-> std::array<
	typename std::decay<
		typename std::common_type<T...>::type
	>::type,
	sizeof...(T)
> {
	return std::array<
		typename std::decay<
			typename std::common_type<T...>::type
		>::type,
		sizeof...(T)
	>{std::forward<T>(values)...};
}

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
//https://stackoverflow.com/questions/21830340/understanding-glmlookat
template<typename real>
_mat<real,4,4> lookAt(
	_vec<real,3> eye,
	_vec<real,3> center,
	_vec<real,3> up
) {
	auto Z = (center - eye).normalize();
	auto Y = up;
	auto X = Y.cross(Z).normalize();
	Y = Z.cross(X);
	return _mat<real,4,4>{
		{X.x, Y.x, -Z.x, -eye.dot(X)},
		{X.y, Y.y, -Z.y, -eye.dot(Y)},
		{X.z, Y.z, -Z.z, -eye.dot(Z)},
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

struct Vertex {
	Tensor::float3 pos;
	Tensor::float3 color;
	Tensor::float3 texCoord;

	static auto getBindingDescription() {
		return VkVertexInputBindingDescription{
			.binding = 0,
			.stride = sizeof(Vertex),
			.inputRate = VK_VERTEX_INPUT_RATE_VERTEX,
		};
	}

	static auto getAttributeDescriptions() {
		return Common::make_array(
			VkVertexInputAttributeDescription{
				.location = 0,
				.binding = 0,
				.format = VK_FORMAT_R32G32B32_SFLOAT,
				.offset = offsetof(Vertex, pos),
			},
			VkVertexInputAttributeDescription{
				.location = 1,
				.binding = 0,
				.format = VK_FORMAT_R32G32B32_SFLOAT,
				.offset = offsetof(Vertex, color),
			},
			VkVertexInputAttributeDescription{
				.location = 2,
				.binding = 0,
				.format = VK_FORMAT_R32G32_SFLOAT,
				.offset = offsetof(Vertex, texCoord),
			}
		);
	}

    bool operator==(Vertex const & o) const {
        return pos == o.pos && color == o.color && texCoord == o.texCoord;
    }
};

namespace std {

template<int dim>
struct hash<Tensor::floatN<dim>> {
	size_t operator()(Tensor::floatN<dim> const & v) const {
		uint32_t h = {};
		for (auto x : v) {
			h ^= hash<uint32_t>()(*(uint32_t const *)&x); 
		}
		return h;
	}
};

template<>
struct hash<Vertex> {
	size_t operator()(Vertex const & v) const {
		return ((hash<Tensor::float3>()(v.pos) ^ (hash<Tensor::float3>()(v.color) << 1)) >> 1) ^ (hash<Tensor::float2>()(v.texCoord) << 1);
	}
};

}

struct UniformBufferObject {
	alignas(16) Tensor::float4x4 model;
	alignas(16) Tensor::float4x4 view;
	alignas(16) Tensor::float4x4 proj;
};
static_assert(sizeof(UniformBufferObject) == 4 * 4 * sizeof(float) * 3);

// why do I think there are already similar classes in vulkan.hpp?

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
	VkAllocationCallbacks const * getAllocator() const { return nullptr; }
};

// custom allocator
struct VulkanAllocator {
	VkAllocationCallbacks * allocator = nullptr;
	VkAllocationCallbacks * getAllocator() { return allocator; }
	VkAllocationCallbacks const * getAllocator() const { return allocator; }
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
	auto operator()() const { return ASSERTHANDLE(handle); }

	VulkanHandle() {} 
	VulkanHandle(Handle handle_) : handle(handle_) {} 
};


struct VulkanInstance : public VulkanHandle<VkInstance> {
	using Super = VulkanHandle<VkInstance>;
	
	~VulkanInstance() {
		if (handle) vkDestroyInstance(handle, getAllocator());
	}

	PFN_vkVoidFunction getProcAddr(char const * const name) const {
		return vkGetInstanceProcAddr((*this)(), name);
	}

	std::vector<VkPhysicalDevice> getPhysicalDevices() const {
		return vulkanEnum<VkPhysicalDevice>(
			NAME_PAIR(vkEnumeratePhysicalDevices),
			(*this)()
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
		auto appInfo = VkApplicationInfo{
			.sType = VK_STRUCTURE_TYPE_APPLICATION_INFO,
			.pApplicationName = title.c_str(),
			.applicationVersion = VK_MAKE_VERSION(1, 0, 0),
			.pEngineName = "No Engine",
			.engineVersion = VK_MAKE_VERSION(1, 0, 0),
			.apiVersion = VK_API_VERSION_1_0,
		};

		// vkCreateInstance needs layerNames
		std::vector<char const *> layerNames;
		if (enableValidationLayers) {
			//insert which of those into our layerName for creating something or something
			//layerNames.push_back("VK_LAYER_LUNARG_standard_validation");	//nope
			layerNames.push_back("VK_LAYER_KHRONOS_validation");	//nope
		}
		
		// vkCreateInstance needs extensions
		auto extensions = getRequiredExtensions(app, enableValidationLayers);

		auto createInfo = VkInstanceCreateInfo{
			.sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO,
			.pApplicationInfo = &appInfo,
			.enabledLayerCount = (uint32_t)layerNames.size(),
			.ppEnabledLayerNames = layerNames.data(),
			.enabledExtensionCount = (uint32_t)extensions.size(),
			.ppEnabledExtensionNames = extensions.data(),
		};
		VULKAN_SAFE(vkCreateInstance, &createInfo, getAllocator(), &handle);
	}
protected:
	std::vector<char const *> getRequiredExtensions(
		::SDLApp::SDLApp const * const app,
		bool const enableValidationLayers
	) {
		// TODO vulkanEnumSDL ?  or just test the return-type for the SDL return type? (assuming it's not the same as the Vulkan return type ...)
		auto extensionCount = uint32_t{};
		SDL_VULKAN_SAFE(SDL_Vulkan_GetInstanceExtensions, app->getWindow(), &extensionCount, nullptr);
		std::vector<char const *> extensions(extensionCount);
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
		
		auto createInfo = VkDebugUtilsMessengerCreateInfoEXT{
			.sType = VK_STRUCTURE_TYPE_DEBUG_UTILS_MESSENGER_CREATE_INFO_EXT,
			.messageSeverity = VK_DEBUG_UTILS_MESSAGE_SEVERITY_VERBOSE_BIT_EXT | VK_DEBUG_UTILS_MESSAGE_SEVERITY_WARNING_BIT_EXT | VK_DEBUG_UTILS_MESSAGE_SEVERITY_ERROR_BIT_EXT,
			.messageType = VK_DEBUG_UTILS_MESSAGE_TYPE_GENERAL_BIT_EXT | VK_DEBUG_UTILS_MESSAGE_TYPE_VALIDATION_BIT_EXT | VK_DEBUG_UTILS_MESSAGE_TYPE_PERFORMANCE_BIT_EXT,
			.pfnUserCallback = debugCallback,
		};
		VULKAN_SAFE(vkCreateDebugUtilsMessengerEXT, (*instance)(), &createInfo, getAllocator(), &handle);
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
	VulkanInstance const * const instance = {};	//from VulkanCommon, needs to be held for dtor to work
public:
	~VulkanSurface() {
		if (handle) vkDestroySurfaceKHR((*instance)(), handle, getAllocator());
	}

	VulkanSurface(
		SDL_Window * const window,
		VulkanInstance const * const instance_
	) : instance(instance_) {
		// https://vulkan-tutorial.com/Drawing_a_triangle/Presentation/Window_surface
		SDL_VULKAN_SAFE(SDL_Vulkan_CreateSurface, window, (*instance)(), &handle);
	}
};

struct VulkanPhysicalDevice : public VulkanHandle<VkPhysicalDevice> {
	using Super = VulkanHandle<VkPhysicalDevice>;
	using Super::Super;

	// TODO maybe I shouldn't return-by-value the structs ...
	
	auto getProperties() const {
		auto physDevProps = VkPhysicalDeviceProperties{};
		vkGetPhysicalDeviceProperties((*this)(), &physDevProps);
		return physDevProps;
	}

	auto getMemoryProperties() const {
		auto memProps = VkPhysicalDeviceMemoryProperties{};
		vkGetPhysicalDeviceMemoryProperties((*this)(), &memProps);
		return memProps;
	}
	
	auto getQueueFamilyProperties() const {
		return vulkanEnum<VkQueueFamilyProperties>(
			NAME_PAIR(vkGetPhysicalDeviceQueueFamilyProperties),
			(*this)()
		);
	}

	auto getExtensionProperties(
		char const * const layerName = nullptr
	) const {
		return vulkanEnum<VkExtensionProperties>(
			NAME_PAIR(vkEnumerateDeviceExtensionProperties),
			(*this)(),
			layerName
		);
	}

	bool getSurfaceSupport(
		uint32_t queueFamilyIndex,
		VkSurfaceKHR surface
	) const {
		VkBool32 presentSupport = VK_FALSE;
		VULKAN_SAFE(vkGetPhysicalDeviceSurfaceSupportKHR, (*this)(), queueFamilyIndex, surface, &presentSupport);
		return !!presentSupport;
	}

	//pass-by-value ok?
	// should these be physical-device-specific or surface-specific?
	//  if a surface needs a physical device ... the latter?
	auto getSurfaceCapabilities(
		VkSurfaceKHR surface
	) const {
		auto caps = VkSurfaceCapabilitiesKHR{};
		VULKAN_SAFE(vkGetPhysicalDeviceSurfaceCapabilitiesKHR, (*this)(), surface, &caps);
		return caps;
	}

	auto getSurfaceFormats(
		VkSurfaceKHR surface
	) const {
		return vulkanEnum<VkSurfaceFormatKHR>(
			NAME_PAIR(vkGetPhysicalDeviceSurfaceFormatsKHR),
			(*this)(),
			surface
		);
	}

	auto getSurfacePresentModes(
		VkSurfaceKHR surface
	) const {
		return vulkanEnum<VkPresentModeKHR>(
			NAME_PAIR(vkGetPhysicalDeviceSurfacePresentModesKHR),
			(*this)(),
			surface
		);
	}

	auto getFeatures() const {
		auto features = VkPhysicalDeviceFeatures{};
		vkGetPhysicalDeviceFeatures((*this)(), &features);
		return features;
	}

	auto createDevice(
		VkDeviceCreateInfo const * const createInfo,
		VkAllocationCallbacks const * const allocator = nullptr//getAllocator()
	) const {
		auto device = VkDevice{};
		VULKAN_SAFE(vkCreateDevice, (*this)(), createInfo, allocator, &device);
		return device;
	}

	auto getFormatProperties(
		VkFormat const format
	) const {
		auto props = VkFormatProperties{};
		vkGetPhysicalDeviceFormatProperties((*this)(), format, &props);
		return props;
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

	auto querySwapChainSupport(
		VkSurfaceKHR surface
	) const {
		return SwapChainSupportDetails{
			.capabilities = getSurfaceCapabilities(surface),
			.formats = getSurfaceFormats(surface),
			.presentModes = getSurfacePresentModes(surface)
		};
	}

	// ************** from here on down, app-specific **************  
protected:
	VkSampleCountFlagBits msaaSamples = VK_SAMPLE_COUNT_1_BIT;
public:
	auto getMSAASamples() const { return msaaSamples; }

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
                msaaSamples = getMaxUsableSampleCount();
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
		auto indices = findQueueFamilies(surface);
		bool extensionsSupported = checkDeviceExtensionSupport(deviceExtensions);
		bool swapChainAdequate = false;
		if (extensionsSupported) {
			auto swapChainSupport = querySwapChainSupport(surface);
			swapChainAdequate = !swapChainSupport.formats.empty() && !swapChainSupport.presentModes.empty();
		}
		VkPhysicalDeviceFeatures features = getFeatures();
		return indices.isComplete()
			&& extensionsSupported
			&& swapChainAdequate
			&& features.samplerAnisotropy;
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

    VkSampleCountFlagBits getMaxUsableSampleCount() const {
        auto physicalDeviceProperties = getProperties();

        auto counts = physicalDeviceProperties.limits.framebufferColorSampleCounts & physicalDeviceProperties.limits.framebufferDepthSampleCounts;
        if (counts & VK_SAMPLE_COUNT_64_BIT) { return VK_SAMPLE_COUNT_64_BIT; }
        if (counts & VK_SAMPLE_COUNT_32_BIT) { return VK_SAMPLE_COUNT_32_BIT; }
        if (counts & VK_SAMPLE_COUNT_16_BIT) { return VK_SAMPLE_COUNT_16_BIT; }
        if (counts & VK_SAMPLE_COUNT_8_BIT) { return VK_SAMPLE_COUNT_8_BIT; }
        if (counts & VK_SAMPLE_COUNT_4_BIT) { return VK_SAMPLE_COUNT_4_BIT; }
        if (counts & VK_SAMPLE_COUNT_2_BIT) { return VK_SAMPLE_COUNT_2_BIT; }
        return VK_SAMPLE_COUNT_1_BIT;
    }

public:
	struct QueueFamilyIndices {
		std::optional<uint32_t> graphicsFamily;
		std::optional<uint32_t> presentFamily;
		bool isComplete() const {
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

public:
    VkFormat findDepthFormat() const {
        return findSupportedFormat(
        	std::vector<VkFormat>{
				VK_FORMAT_D32_SFLOAT,
				VK_FORMAT_D32_SFLOAT_S8_UINT,
				VK_FORMAT_D24_UNORM_S8_UINT
			},
            VK_IMAGE_TILING_OPTIMAL,
            VK_FORMAT_FEATURE_DEPTH_STENCIL_ATTACHMENT_BIT
        );
    }
protected: 
	VkFormat findSupportedFormat(
		std::vector<VkFormat> const & candidates,
		VkImageTiling const tiling,
		VkFormatFeatureFlags const features
	) const {
        for (auto format : candidates) {
            auto props = getFormatProperties(format);
            if (tiling == VK_IMAGE_TILING_LINEAR && 
				(props.linearTilingFeatures & features) == features
			) {
                return format;
            } else if (tiling == VK_IMAGE_TILING_OPTIMAL && 
				(props.optimalTilingFeatures & features) == features
			) {
                return format;
            }
        }
        throw Common::Exception() << "failed to find supported format!";
    }

};

// validationLayers matches in checkValidationLayerSupport and initLogicalDevice
std::vector<char const *> validationLayers = {
	"VK_LAYER_KHRONOS_validation"
};

struct VulkanQueue : public VulkanHandle<VkQueue> {
	using Super = VulkanHandle<VkQueue>;
	using Super::Super;
	
	void waitIdle() const {
		VULKAN_SAFE(vkQueueWaitIdle, (*this)());
	}

	// see, no pass-by-copy.
	// this class is a harmless wrapper so far.
	// in fact (hmm) should I even throw results or should I return them?  100% wrapper.
	void submit(
		uint32_t submitCount,
		VkSubmitInfo const * const pSubmits,
		VkFence fence = VK_NULL_HANDLE
	) const {
		VULKAN_SAFE(vkQueueSubmit, handle, submitCount, pSubmits, fence);
	}

	VkResult present(
		VkPresentInfoKHR const * const pPresentInfo
	) const {
		return vkQueuePresentKHR(handle, pPresentInfo);
	}
};

struct VulkanDevice : public VulkanHandle<VkDevice> {
protected:
	std::unique_ptr<VulkanQueue> graphicsQueue;
	std::unique_ptr<VulkanQueue> presentQueue;
public:
	VulkanQueue const * getGraphicsQueue() const { return graphicsQueue.get(); }
	VulkanQueue const * getPresentQueue() const { return presentQueue.get(); }
	
	~VulkanDevice() {
		if (handle) vkDestroyDevice(handle, getAllocator());
	}
	
	// should this return a handle or an object?
	// I'll return a handle like the create*** functions
	VulkanQueue getQueue(
		uint32_t queueFamilyIndex,
		uint32_t queueIndex = 0
	) const {
		auto result = VkQueue{};
		// should this call the getter or should VulkanQueue's ctor?
		vkGetDeviceQueue((*this)(), queueFamilyIndex, queueIndex, &result);
		return result;
	}
	
	// maybe there's no need for these 'create' functions 

#define CREATE_CREATER(name, suffix)\
	Vk##name##suffix create##name(\
		Vk##name##CreateInfo##suffix const * const createInfo,\
		VkAllocationCallbacks const * const allocator = nullptr/*getAllocator()*/\
	) const {\
		auto result = Vk##name##suffix{};\
		VULKAN_SAFE(vkCreate##name##suffix, (*this)(), createInfo, allocator, &result);\
		return result;\
	}

	// meanwhile all these return handles, not wrappers.
CREATE_CREATER(Swapchain, KHR)
CREATE_CREATER(RenderPass, )
CREATE_CREATER(Buffer, )
CREATE_CREATER(Sampler, )
CREATE_CREATER(Image, )
CREATE_CREATER(ImageView, )
CREATE_CREATER(Framebuffer, )
CREATE_CREATER(DescriptorSetLayout, )
CREATE_CREATER(ShaderModule, )
CREATE_CREATER(PipelineLayout, )
CREATE_CREATER(CommandPool, )

	VkPipeline createGraphicsPipelines(
		VkPipelineCache pipelineCache,
		size_t numCreateInfo,
		VkGraphicsPipelineCreateInfo const * const createInfo,
		VkAllocationCallbacks const * const allocator = nullptr//getAllocator()
	) const {
		auto result = VkPipeline{};
		VULKAN_SAFE(vkCreateGraphicsPipelines, (*this)(), pipelineCache, numCreateInfo, createInfo, allocator, &result);
		return result;
	}
	
	void waitIdle() const {
		VULKAN_SAFE(vkDeviceWaitIdle, (*this)());
	}

	// ************** from here on down, app-specific **************  
	
	VulkanDevice(
		VulkanPhysicalDevice const * const physicalDevice,
		VulkanSurface const * const surface,
		std::vector<char const *> const & deviceExtensions,
		bool enableValidationLayers
	) {
		auto indices = physicalDevice->findQueueFamilies((*surface)());

		auto queuePriorities = Common::make_array<float>(1);
		std::vector<VkDeviceQueueCreateInfo> queueCreateInfos;
		for (uint32_t queueFamily : std::set<uint32_t>{
			indices.graphicsFamily.value(),
			indices.presentFamily.value(),
		}) {
			queueCreateInfos.push_back(VkDeviceQueueCreateInfo{
				.sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO,
				.queueFamilyIndex = queueFamily,
				.queueCount = (uint32_t)queuePriorities.size(),
				.pQueuePriorities = queuePriorities.data(),
			});
		}

		auto deviceFeatures = VkPhysicalDeviceFeatures{
			.samplerAnisotropy = VK_TRUE,
		};

		auto createInfo = VkDeviceCreateInfo{
			.sType = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO,
			.queueCreateInfoCount = (uint32_t)queueCreateInfos.size(),
			.pQueueCreateInfos = queueCreateInfos.data(),
			.enabledExtensionCount = (uint32_t)deviceExtensions.size(),
			.ppEnabledExtensionNames = deviceExtensions.data(),
			.pEnabledFeatures = &deviceFeatures,
		};
		if (enableValidationLayers) {
			createInfo.enabledLayerCount = (uint32_t)validationLayers.size();
			createInfo.ppEnabledLayerNames = validationLayers.data();
		} else {
			createInfo.enabledLayerCount = 0;
		}
		handle = physicalDevice->createDevice(&createInfo);
	
		graphicsQueue = std::make_unique<VulkanQueue>(getQueue(indices.graphicsFamily.value()));
		presentQueue = std::make_unique<VulkanQueue>(getQueue(indices.presentFamily.value()));
	}
};

struct VulkanRenderPass : public VulkanHandle<VkRenderPass> {
protected:
	//held
	VulkanDevice const * const device = {};
public:
	~VulkanRenderPass() {
		if (handle) vkDestroyRenderPass((*device)(), handle, getAllocator());
	}
	
	// ************** from here on down, app-specific **************  

	VulkanRenderPass(
		VulkanPhysicalDevice const * const physicalDevice,
		VulkanDevice const * const device_,
		VkFormat swapChainImageFormat
	) : device(device_) {
		auto attachments = Common::make_array(
			VkAttachmentDescription{	//colorAttachment
				.format = swapChainImageFormat,
				.samples = physicalDevice->getMSAASamples(),
				.loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR,
				.storeOp = VK_ATTACHMENT_STORE_OP_STORE,
				.stencilLoadOp = VK_ATTACHMENT_LOAD_OP_DONT_CARE,
				.stencilStoreOp = VK_ATTACHMENT_STORE_OP_DONT_CARE,
				.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED,
				.finalLayout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL,
			},
			VkAttachmentDescription{	//depthAttachment
				.format = physicalDevice->findDepthFormat(),
				.samples = physicalDevice->getMSAASamples(),
				.loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR,
				.storeOp = VK_ATTACHMENT_STORE_OP_DONT_CARE,
				.stencilLoadOp = VK_ATTACHMENT_LOAD_OP_DONT_CARE,
				.stencilStoreOp = VK_ATTACHMENT_STORE_OP_DONT_CARE,
				.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED,
				.finalLayout = VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL,
			},
			VkAttachmentDescription{	//colorAttachmentResolve
				.format = swapChainImageFormat,
				.samples = VK_SAMPLE_COUNT_1_BIT,
				.loadOp = VK_ATTACHMENT_LOAD_OP_DONT_CARE,
				.storeOp = VK_ATTACHMENT_STORE_OP_STORE,
				.stencilLoadOp = VK_ATTACHMENT_LOAD_OP_DONT_CARE,
				.stencilStoreOp = VK_ATTACHMENT_STORE_OP_DONT_CARE,
				.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED,
				.finalLayout = VK_IMAGE_LAYOUT_PRESENT_SRC_KHR,
			}
		);
		auto colorAttachmentRef = VkAttachmentReference{
			.attachment = 0,
			.layout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL,
		};
		auto depthAttachmentRef = VkAttachmentReference{
			.attachment = 1,
			.layout = VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL,
		};	
        auto colorAttachmentResolveRef = VkAttachmentReference{
			.attachment = 2,
			.layout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL,
		};
		auto subpasses = Common::make_array(
			VkSubpassDescription{
				.pipelineBindPoint = VK_PIPELINE_BIND_POINT_GRAPHICS,
				.colorAttachmentCount = 1,
				.pColorAttachments = &colorAttachmentRef,
        		.pResolveAttachments = &colorAttachmentResolveRef,
				.pDepthStencilAttachment = &depthAttachmentRef,
			}
		);
		auto dependencies = Common::make_array(
			VkSubpassDependency{
				.srcSubpass = VK_SUBPASS_EXTERNAL,
				.dstSubpass = 0,
				.srcStageMask = 
					VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT | 
					VK_PIPELINE_STAGE_EARLY_FRAGMENT_TESTS_BIT,
				.dstStageMask = 
					VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT | 
					VK_PIPELINE_STAGE_EARLY_FRAGMENT_TESTS_BIT,
				.srcAccessMask = 0,
				.dstAccessMask = 
					VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT | 
					VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT,
			}
		);
		auto renderPassInfo = VkRenderPassCreateInfo{
			.sType = VK_STRUCTURE_TYPE_RENDER_PASS_CREATE_INFO,
			.attachmentCount = (uint32_t)attachments.size(),
			.pAttachments = attachments.data(),
			.subpassCount = (uint32_t)subpasses.size(),
			.pSubpasses = subpasses.data(),
			.dependencyCount = (uint32_t)dependencies.size(),
			.pDependencies = dependencies.data(),
		};
		handle = device_->createRenderPass(&renderPassInfo);
	}
};

struct VulkanBuffer : public VulkanHandle<VkBuffer> {
	using Super = VulkanHandle<VkBuffer>;
protected:
	//holds
	VulkanDevice const * device = {};
public:
	~VulkanBuffer() {
		if (handle) vkDestroyBuffer((*device)(), handle, getAllocator());
	}

	VulkanBuffer(
		VkBuffer handle_,
		VulkanDevice const * const device_
	) : Super(handle_),
		device(device_)
	{}

	VulkanBuffer(
		VulkanDevice const * const device_,
		VkBufferCreateInfo const createInfo
	) : Super(device_->createBuffer(&createInfo, getAllocator())),
		device(device_)
	{}
};

/*
ok i've got 
- VkBuffer with VkDeviceMemory
- VkImage with VkDeviceMemory
- VkImage with VkImageView (and VkFramebuffer?)
*/
struct VulkanImage : public VulkanHandle<VkImage> {
	using Super = VulkanHandle<VkImage>;
protected:
	//holds
	VulkanDevice const * const device = {};
public:
	~VulkanImage() {
		if (handle) vkDestroyImage((*device)(), handle, getAllocator());
	}

	VulkanImage(
		VkImage handle_,
		VulkanDevice const * const device_
	) : Super(handle_),
		device(device_)
	{}

	VulkanImage(
		VulkanDevice const * const device_,
		VkImageCreateInfo const createInfo
	) : Super(device_->createImage(&createInfo, getAllocator())),
		device(device_)
	{}
};

struct VulkanImageView : public VulkanHandle<VkImageView> {
	using Super = VulkanHandle<VkImageView>;
protected:
	//holds
	VulkanDevice const * device = {};
public:
	~VulkanImageView() {
		if (handle) vkDestroyImageView((*device)(), handle, getAllocator());
	}

	//should I put handle first since all subclasses of VulkanHandle have handle-first?
	// (but not all have device-first)
	//or should I put device first since all ctors of VulkanImageView have device required (but not necessarily handle) ?
	VulkanImageView(
		VkImageView handle_,
		VulkanDevice const * const device_
	) : Super(handle_),
		device(device_)
	{}

	VulkanImageView(
		VulkanDevice const * const device_,
		VkImageViewCreateInfo const createInfo
	) : Super(device_->createImageView(&createInfo, getAllocator())),
		device(device_) 
	{}
};

struct VulkanSampler : public VulkanHandle<VkSampler> {
	using Super = VulkanHandle<VkSampler>;
protected:
	//holds
	VulkanDevice const * device = {};
public:
	~VulkanSampler() {
		if (handle) vkDestroySampler((*device)(), handle, getAllocator());
	}

	VulkanSampler(
		VkSampler handle_,
		VulkanDevice const * const device_
	) : Super(handle_),
		device(device_)
	{}

	VulkanSampler(
		VulkanDevice const * const device_,
		VkSamplerCreateInfo const createInfo
	) : Super(device_->createSampler(&createInfo, getAllocator())),
		device(device_)
	{}
};

struct VulkanCommandBuffer : public VulkanHandle<VkCommandBuffer> {
	using Super = VulkanHandle<VkCommandBuffer>;
protected:
	VulkanDevice const * const device = {};
	VkCommandPool commandPool = {};
public:
	~VulkanCommandBuffer() {
		if (handle) vkFreeCommandBuffers((*device)(), commandPool, 1, &handle);
	}

	VulkanCommandBuffer(
		VkCommandBuffer const handle_,
		VulkanDevice const * const device_,
		VkCommandPool const commandPool_
	) : Super(handle_),
		device(device_),
		commandPool(commandPool_)
	{}

	void copyBuffer(
		VkBuffer const srcBuffer,
		VkBuffer const dstBuffer,
		VkBufferCopy const region
	) const {
		vkCmdCopyBuffer(
			(*this)(),
			srcBuffer,
			dstBuffer,
			1,	//count ... TODO make an array-based one?
			&region);
	}

	void copyBufferToImage(
		VkBuffer const buffer,
		VkImage const image,
		uint32_t width,
		uint32_t height,
		VkBufferImageCopy const region
	) const {
		vkCmdCopyBufferToImage(
			(*this)(),
			buffer,
			image,
			VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
			1,
			&region
		);
	}

#if 0	
	// TODO but dont use vector args cuz thats needless allocs
	// try to do array args, but that makes empty arrays {} fail to type-deduce ...
	// hmm another way is to template a tuple, then lambda iterate across it, 
	//  and fill out memBarriers, bufferMemBarriers, imageMemBarriers according to the types found ...
	// might be cumbersome ...
	void pipelineBarrier(
		VkPipelineStageFlags const srcStageMask,
		VkPipelineStageFlags const dstStageMask,
		VkDependencyFlags const dependencyFlags,
		std::array<VkMemoryBarrier> memBarriers,
		std::array<VkBufferMemoryBarrier> bufferMemBarriers,
		std::array<VkImageMemoryBarrier> imageMemBarriers
	) const {
		vkCmdPipelineBarrier(
			(*this)(),
			srcStageMask,
			dstStageMask,
			dependencyFlags,
			(uint32_t)memBarriers.size(),
			memBarriers.data(),
			(uint32_t)bufferMemBarriers.size(),
			bufferMemBarriers.data(),
			(uint32_t)imageMemBarriers.size(),
			imageMemBarriers.data()
		);
	}
#endif
};

struct VulkanSingleTimeCommand : public VulkanCommandBuffer {
	using Super = VulkanCommandBuffer;
	VulkanSingleTimeCommand(
		VulkanDevice const * const device_,
		VkCommandPool commandPool_
	) : Super({}, device_, commandPool_) 
	{
		// TODO this matches 'initCommandBuffers' for each frame-in-flight
		auto allocInfo = VkCommandBufferAllocateInfo{
			.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO,
			.commandPool = commandPool,
			.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY,
			.commandBufferCount = 1,
		};
		vkAllocateCommandBuffers((*device)(), &allocInfo, &handle);
		// end part that matches
		// and this part kinda matches the start of 'recordCommandBuffer'
		auto beginInfo = VkCommandBufferBeginInfo{
			.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO,
			.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT,
		};
		VULKAN_SAFE(vkBeginCommandBuffer, handle, &beginInfo);
		//end part that matches
	}
	
	~VulkanSingleTimeCommand() {
		VULKAN_SAFE(vkEndCommandBuffer, handle);
		auto submitInfo = VkSubmitInfo{
			.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO,
			.commandBufferCount = 1,
			.pCommandBuffers = &handle,
		};
		auto queue = device->getGraphicsQueue();
		queue->submit(1, &submitInfo);
		queue->waitIdle();
	}
};

struct VulkanCommandPool : public VulkanHandle<VkCommandPool> {
protected:
	//held:
	VulkanDevice const * const device = {};
public:
	~VulkanCommandPool() {
		if (handle) vkDestroyCommandPool((*device)(), handle, getAllocator());
	}
	VulkanCommandPool(
		VulkanPhysicalDevice const * const physicalDevice,
		VulkanDevice const * const device_,
		VkSurfaceKHR surface
	) : device(device_) {
		auto queueFamilyIndices = physicalDevice->findQueueFamilies(surface);
		auto poolInfo = VkCommandPoolCreateInfo{
			.sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO,
			.flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT,
			.queueFamilyIndex = queueFamilyIndices.graphicsFamily.value(),
		};
		handle = device_->createCommandPool(&poolInfo);
	}

	//copies based on the graphicsQueue
	// used by makeBufferFromStaged
	void copyBuffer(
		VkBuffer srcBuffer,	//staging VkBuffer
		VkBuffer dstBuffer,	//dest VkBuffer
		VkDeviceSize size
	) const {
		VulkanSingleTimeCommand(device, (*this)())
		.copyBuffer(
			srcBuffer,
			dstBuffer,
			VkBufferCopy{
				.size = size,
			}
		);
	}

	void copyBufferToImage(
		VkBuffer buffer,
		VkImage image,
		uint32_t width,
		uint32_t height
	) const {
		VulkanSingleTimeCommand(device, (*this)())
		.copyBufferToImage(
			buffer,
			image,
			width,
			height,
			VkBufferImageCopy{
				.bufferOffset = 0,
				.bufferRowLength = 0,
				.bufferImageHeight = 0,
				.imageSubresource = {
					.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT,
					.mipLevel = 0,
					.baseArrayLayer = 0,
					.layerCount = 1,
				},
				.imageOffset = {0, 0, 0},
				.imageExtent = {
					width,
					height,
					1
				},
			}	
		);
	}

	void transitionImageLayout(
		VkImage image,
		VkFormat format,
		VkImageLayout oldLayout,
		VkImageLayout newLayout,
		uint32_t mipLevels
	) const {
		VulkanSingleTimeCommand commandBuffer(device, (*this)());

		auto barrier = VkImageMemoryBarrier{
			.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER,
			.oldLayout = oldLayout,
			.newLayout = newLayout,
			.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
			.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
			.image = image,
			.subresourceRange = {
				.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT,
				.baseMipLevel = 0,
				.levelCount = mipLevels,
				.baseArrayLayer = 0,
				.layerCount = 1,
			},
		};

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
			commandBuffer(),
			sourceStage,
			destinationStage,
			0,
			0,
			nullptr,
			0,
			nullptr,
			1,
			&barrier
		);
	}
};

// methods common to VkBuffer and VkImage paired with VkDeviceMemory
template<typename Super_>
struct VulkanDeviceMemoryParent  : public Super_ {
	using Super = Super_;
protected:
	//owns
	VkDeviceMemory memory = {};
	//holds
	VulkanDevice const * device = {};
public:
	auto getMemory() const { return ASSERTHANDLE(memory); }

	~VulkanDeviceMemoryParent() {
		if (memory) vkFreeMemory((*device)(), memory, Super::getAllocator());
		//doesn't destroy handle -- that's for the child class to do
	}

	VulkanDeviceMemoryParent(
		Super::Handle handle_,
		VkDeviceMemory memory_,
		VulkanDevice const * const device_
	) : Super(handle_, device_),
		memory(memory_),
		device(device_)
	{}

	//only call this after handle is filled
	// is this a 'device' method or a 'buffer' method?
	// is this only for VkBuffer or also for VkImage ?
	VkMemoryRequirements getMemoryRequirements() const {
		auto memRequirements = VkMemoryRequirements{};
		vkGetBufferMemoryRequirements((*device)(), (*this)(), &memRequirements);
		return memRequirements;
	}
};

struct VulkanDeviceMemoryBuffer : public VulkanDeviceMemoryParent<VulkanBuffer> {
	using Super = VulkanDeviceMemoryParent<VulkanBuffer>;
	using Super::Super;

	//ctor for VkBuffer's whether they are being ctor'd by staging or by uniforms whatever
	VulkanDeviceMemoryBuffer(
		VulkanPhysicalDevice const * const physicalDevice,
		VulkanDevice const * const device_,
		VkDeviceSize size,
		VkBufferUsageFlags usage,
		VkMemoryPropertyFlags properties
	) 
#if 0	// new
// TODO get	VulkanDeviceMemoryParent to pass-thru correctly
	: Super(
		device_,
		VkBufferCreateInfo{
			.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO,
			.size = size,
			.usage = usage,
			.sharingMode = VK_SHARING_MODE_EXCLUSIVE,
		}
	) {
#else 	//old
	: Super({}, {}, device_) {
		auto bufferInfo = VkBufferCreateInfo{
			.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO,
			.size = size,
			.usage = usage,
			.sharingMode = VK_SHARING_MODE_EXCLUSIVE,
		};
		Super::handle = device_->createBuffer(&bufferInfo);
#endif

		VkMemoryRequirements memRequirements = Super::getMemoryRequirements();
		auto allocInfo = VkMemoryAllocateInfo{
			.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO,
			.allocationSize = memRequirements.size,
			.memoryTypeIndex = physicalDevice->findMemoryType(memRequirements.memoryTypeBits, properties),
		};
		VULKAN_SAFE(vkAllocateMemory, (*Super::device)(), &allocInfo, getAllocator(), &memory);

		vkBindBufferMemory((*Super::device)(), (*this)(), Super::memory, 0);
	}

	// TODO make a StagingBuffer subclass? 
	static std::unique_ptr<VulkanDeviceMemoryBuffer> makeFromStaged(
		VulkanPhysicalDevice const * const physicalDevice,
		VulkanDevice const * const device,
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
	static std::unique_ptr<VulkanDeviceMemoryBuffer> makeBufferFromStaged(
		VulkanPhysicalDevice const * const physicalDevice,
		VulkanDevice const * const device,
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
		
		commandPool->copyBuffer(
			(*stagingBuffer)(),
			(*buffer)(),
			bufferSize
		);
	
		return buffer;
	}
};

struct VulkanDeviceMemoryImage : public VulkanDeviceMemoryParent<VulkanImage> {
	using Super = VulkanDeviceMemoryParent<VulkanImage>;
	using Super::Super;

public:
	static std::unique_ptr<VulkanDeviceMemoryImage> makeTextureFromStaged(
		VulkanPhysicalDevice const * const physicalDevice,
		VulkanDevice const * const device,
		VulkanCommandPool const * const commandPool,
		void const * const srcData,
		size_t bufferSize,
		int texWidth,
		int texHeight,
		uint32_t mipLevels
	) {
		std::unique_ptr<VulkanDeviceMemoryBuffer> stagingBuffer = VulkanDeviceMemoryBuffer::makeFromStaged(
			physicalDevice,
			device,
			srcData,
			bufferSize
		);
		
		auto image = createImage(
			physicalDevice,
			device,
			texWidth,
			texHeight,
			mipLevels,
			VK_SAMPLE_COUNT_1_BIT,
			VK_FORMAT_R8G8B8A8_SRGB,
			VK_IMAGE_TILING_OPTIMAL,
			VK_IMAGE_USAGE_TRANSFER_SRC_BIT 
			| VK_IMAGE_USAGE_TRANSFER_DST_BIT 
			| VK_IMAGE_USAGE_SAMPLED_BIT,
			VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT
		);

		commandPool->transitionImageLayout(
			(*image)(),
			VK_FORMAT_R8G8B8A8_SRGB,
			VK_IMAGE_LAYOUT_UNDEFINED,
			VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
			mipLevels
		);
		commandPool->copyBufferToImage(
			(*stagingBuffer)(),
			(*image)(),
			(uint32_t)texWidth,
			(uint32_t)texHeight
		);
		/*
		commandPool->transitionImageLayout(
			(*image)(),
			VK_FORMAT_R8G8B8A8_SRGB,
			VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
			VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL
		);
		*/
		
		return image;
	}

public:
	static std::unique_ptr<VulkanDeviceMemoryImage> createImage(
		VulkanPhysicalDevice const * const physicalDevice,
		VulkanDevice const * const device,
		uint32_t width,
		uint32_t height,
		uint32_t mipLevels,
		VkSampleCountFlagBits numSamples,
		VkFormat format,
		VkImageTiling tiling,
		VkImageUsageFlags usage,
		VkMemoryPropertyFlags properties
	) {
#if 1
		auto image = VkImage{};
		auto imageInfo = VkImageCreateInfo{
			.sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO,
			.imageType = VK_IMAGE_TYPE_2D,
			.format = format,
			.extent = {
				.width = width,
				.height = height,
				.depth = 1,
			},
			.mipLevels = mipLevels,
			.arrayLayers = 1,
			.samples = numSamples,
			.tiling = tiling,
			.usage = usage,
			.sharingMode = VK_SHARING_MODE_EXCLUSIVE,
			.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED,
		};
		VULKAN_SAFE(vkCreateImage, (*device)(), &imageInfo, nullptr/*getAllocator()*/, &image);
#else		
		// TODO this as a ctor that just calls Super
		VulkanImage image(
			device,
			VkImageCreateInfo{
				.sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO,
				.imageType = VK_IMAGE_TYPE_2D,
				.format = format,
				.extent = {
					.width = width,
					.height = height,
					.depth = 1,
				},
				.mipLevels = mipLevels,
				.arrayLayers = 1,
				.samples = numSamples,
				.tiling = tiling,
				.usage = usage,
				.sharingMode = VK_SHARING_MODE_EXCLUSIVE,
				.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED,
			}
		);
#endif
		auto memRequirements = VkMemoryRequirements{};
		vkGetImageMemoryRequirements((*device)(), image, &memRequirements);

		auto imageMemory = VkDeviceMemory{};
		auto allocInfo = VkMemoryAllocateInfo{
			.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO,
			.allocationSize = memRequirements.size,
			.memoryTypeIndex = physicalDevice->findMemoryType(memRequirements.memoryTypeBits, properties),
		};
		VULKAN_SAFE(vkAllocateMemory, (*device)(), &allocInfo, nullptr/*getAllocator()*/, &imageMemory);

		vkBindImageMemory((*device)(), image, imageMemory, 0);
		
		return std::make_unique<VulkanDeviceMemoryImage>(image, imageMemory, device);
	}
};

struct VulkanSwapChain : public VulkanHandle<VkSwapchainKHR> {
protected:
	//owned
	std::unique_ptr<VulkanRenderPass> renderPass;
	// hold for this class lifespan
	VulkanDevice const * const device = {};

    std::unique_ptr<VulkanDeviceMemoryImage> depthImage;
    std::unique_ptr<VulkanImageView> depthImageView;
    
	std::unique_ptr<VulkanDeviceMemoryImage> colorImage;
    std::unique_ptr<VulkanImageView> colorImageView;
public:
	VkExtent2D extent = {};
	
	// I would combine these into one struct so they can be dtored together
	// but it seems vulkan wants VkImages linear for its getter?
	std::vector<VkImage> images;
	std::vector<std::unique_ptr<VulkanImageView>> imageViews;
	std::vector<VkFramebuffer> framebuffers;
	
public:
	VulkanRenderPass const * const getRenderPass() const {
		return renderPass.get();
	}

	~VulkanSwapChain() {
        depthImageView = nullptr;
        depthImage = nullptr;
		colorImageView = nullptr;
        colorImage = nullptr;
		
		for (auto framebuffer : framebuffers) {
			vkDestroyFramebuffer((*device)(), framebuffer, getAllocator());
		}
		renderPass = nullptr;
		imageViews.clear();
		if (handle) vkDestroySwapchainKHR((*device)(), handle, getAllocator());
	}

	// should this be a 'devices' or a 'swapchain' method?
	std::vector<VkImage> getImages() const {
		return vulkanEnum<VkImage>(NAME_PAIR(vkGetSwapchainImagesKHR), (*device)(), (*this)());
	}
	
	// ************** from here on down, app-specific **************  
	
	VulkanSwapChain(
		Tensor::int2 screenSize,
		VulkanPhysicalDevice const * const physicalDevice,
		VulkanDevice const * const device_,
		VulkanSurface const * const surface
	) : device(device_) {
		auto swapChainSupport = physicalDevice->querySwapChainSupport((*surface)());
		auto surfaceFormat = chooseSwapSurfaceFormat(swapChainSupport.formats);
		auto presentMode = chooseSwapPresentMode(swapChainSupport.presentModes);
		extent = chooseSwapExtent(screenSize, swapChainSupport.capabilities);

		// how come imageCount is one less than vkGetSwapchainImagesKHR gives?
		// maxImageCount == 0 means no max?
		uint32_t imageCount = swapChainSupport.capabilities.minImageCount + 1;
		if (swapChainSupport.capabilities.maxImageCount > 0) {
			imageCount = std::min(imageCount, swapChainSupport.capabilities.maxImageCount);
		}

		auto createInfo = VkSwapchainCreateInfoKHR{
			.sType = VK_STRUCTURE_TYPE_SWAPCHAIN_CREATE_INFO_KHR,
			.surface = (*surface)(),
			.minImageCount = imageCount,
			.imageFormat = surfaceFormat.format,
			.imageColorSpace = surfaceFormat.colorSpace,
			.imageExtent = extent,
			.imageArrayLayers = 1,
			.imageUsage = VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT,
			.preTransform = swapChainSupport.capabilities.currentTransform,
			.compositeAlpha = VK_COMPOSITE_ALPHA_OPAQUE_BIT_KHR,
			.presentMode = presentMode,
			.clipped = VK_TRUE,
		};
		auto indices = physicalDevice->findQueueFamilies((*surface)());
		auto queueFamilyIndices = Common::make_array<uint32_t>(
			(uint32_t)indices.graphicsFamily.value(),
			(uint32_t)indices.presentFamily.value()
		);
		if (indices.graphicsFamily != indices.presentFamily) {
			createInfo.imageSharingMode = VK_SHARING_MODE_CONCURRENT;
			createInfo.queueFamilyIndexCount = (uint32_t)queueFamilyIndices.size();
			createInfo.pQueueFamilyIndices = queueFamilyIndices.data();
		} else {
			createInfo.imageSharingMode = VK_SHARING_MODE_EXCLUSIVE;
		}
		handle = device_->createSwapchain(&createInfo);

		images = getImages();
		for (size_t i = 0; i < images.size(); i++) {
			imageViews.push_back(createImageView(
				images[i],
				surfaceFormat.format,
				VK_IMAGE_ASPECT_COLOR_BIT,
				1
			));
		}
	
		renderPass = std::make_unique<VulkanRenderPass>(physicalDevice, device_, surfaceFormat.format);
        
		//createColorResources
        VkFormat colorFormat = surfaceFormat.format;
        colorImage = VulkanDeviceMemoryImage::createImage(
			physicalDevice,
			device,
			extent.width,
			extent.height,
			1,
			physicalDevice->getMSAASamples(),
			colorFormat,
			VK_IMAGE_TILING_OPTIMAL,
			VK_IMAGE_USAGE_TRANSIENT_ATTACHMENT_BIT | VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT,
			VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT
		);
        colorImageView = createImageView(
			(*colorImage)(),
			colorFormat,
			VK_IMAGE_ASPECT_COLOR_BIT,
			1
		);
		
		//createDepthResources
        VkFormat depthFormat = physicalDevice->findDepthFormat();
        depthImage = VulkanDeviceMemoryImage::createImage(
			physicalDevice,
			device,
			extent.width,
			extent.height,
			1,
			physicalDevice->getMSAASamples(),
			depthFormat,
			VK_IMAGE_TILING_OPTIMAL,
			VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT,
			VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT
		);
        depthImageView = createImageView(
			(*depthImage)(),
			depthFormat,
			VK_IMAGE_ASPECT_DEPTH_BIT,
			1
		);
		
		//createFramebuffers
		framebuffers.resize(imageViews.size());
		for (size_t i = 0; i < imageViews.size(); i++) {
			auto attachments = Common::make_array(
				(*colorImageView)(),
				(*depthImageView)(),
				(*imageViews[i])()
			);
			auto framebufferInfo = VkFramebufferCreateInfo{
				.sType = VK_STRUCTURE_TYPE_FRAMEBUFFER_CREATE_INFO,
				.renderPass = (*renderPass)(),
				.attachmentCount = (uint32_t)attachments.size(),
				.pAttachments = attachments.data(),
				.width = extent.width,
				.height = extent.height,
				.layers = 1,
			};
			framebuffers[i] = device_->createFramebuffer(&framebufferInfo);
		}
	}
public:
	std::unique_ptr<VulkanImageView> createImageView(
		VkImage image,
		VkFormat format,
		VkImageAspectFlags aspectFlags,
		uint32_t mipLevels
	) {
		return std::make_unique<VulkanImageView>(
			device,
			VkImageViewCreateInfo{
				.sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO,
				.image = image,
				.viewType = VK_IMAGE_VIEW_TYPE_2D,
				.format = format,
				.subresourceRange = {
					.aspectMask = aspectFlags,
					.baseMipLevel = 0,
					.levelCount = mipLevels,
					.baseArrayLayer = 0,
					.layerCount = 1,
				},
			}
		);
	}

protected:
	static VkSurfaceFormatKHR chooseSwapSurfaceFormat(
		std::vector<VkSurfaceFormatKHR> const & availableFormats
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
		std::vector<VkPresentModeKHR> const & availablePresentModes
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
				(uint32_t)screenSize.x,
				(uint32_t)screenSize.y
			};
			actualExtent.width = std::clamp(actualExtent.width, capabilities.minImageExtent.width, capabilities.maxImageExtent.width);
			actualExtent.height = std::clamp(actualExtent.height, capabilities.minImageExtent.height, capabilities.maxImageExtent.height);
			return actualExtent;
		}
	}
};

//only used by VulkanGraphicsPipeline
struct VulkanDescriptorSetLayout : public VulkanHandle<VkDescriptorSetLayout> {
	using Super = VulkanHandle<VkDescriptorSetLayout>;
protected:
	//held for dtor
	VulkanDevice const * const device = {};
public:
	~VulkanDescriptorSetLayout() {
		if (handle) {
			vkDestroyDescriptorSetLayout((*device)(), handle, getAllocator());
		}
	}
	
	VulkanDescriptorSetLayout(
		VulkanDevice const * const device_
	) : device(device_) {
		auto bindings = Common::make_array(
			VkDescriptorSetLayoutBinding{	//uboLayoutBinding 
				.binding = 0,
				.descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER,
				.descriptorCount = 1,
				.stageFlags = VK_SHADER_STAGE_VERTEX_BIT,
			},
			VkDescriptorSetLayoutBinding{	//samplerLayoutBinding 
				.binding = 1,
				.descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
				.descriptorCount = 1,
				.stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT,
			}
		);
		auto layoutInfo = VkDescriptorSetLayoutCreateInfo{
			.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO,
			.bindingCount = (uint32_t)bindings.size(),
			.pBindings = bindings.data(),
		};
		handle = device_->createDescriptorSetLayout(&layoutInfo);
	}
};

struct VulkanDescriptorPool : public VulkanHandle<VkDescriptorPool> {
	using Super = VulkanHandle<VkDescriptorPool>;
protected:
	//held for dtor
	VulkanDevice const * const device = {};
public:
	~VulkanDescriptorPool() {
		if (handle) vkDestroyDescriptorPool((*device)(), handle, getAllocator());
	}

	VulkanDescriptorPool(
		VulkanDevice const * const device_,
		uint32_t const maxFramesInFlight
	) : device(device_) {
		auto poolSizes = Common::make_array(
			VkDescriptorPoolSize{
				.type = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER,
				.descriptorCount = maxFramesInFlight,
			},
			VkDescriptorPoolSize{
				.type = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
				.descriptorCount = maxFramesInFlight,
			}
		);
		auto poolInfo = VkDescriptorPoolCreateInfo{
			.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO,
			.maxSets = maxFramesInFlight,
			.poolSizeCount = (uint32_t)poolSizes.size(),
			.pPoolSizes = poolSizes.data(),
		};
		VULKAN_SAFE(vkCreateDescriptorPool, (*device)(), &poolInfo, getAllocator(), &handle);
	}
};

//only used by VulkanGraphicsPipeline's ctor
struct VulkanShaderModule : public VulkanHandle<VkShaderModule> {
protected:
	//held:
	VulkanDevice const * const device = {};
public:
	~VulkanShaderModule() {
		if (handle) vkDestroyShaderModule((*device)(), handle, getAllocator());
	}
	
	VulkanShaderModule(
		VulkanDevice const * const device_,
		std::string const code
	) : device(device_) {
		auto createInfo = VkShaderModuleCreateInfo{
			.sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO,
			.codeSize = code.length(),
			.pCode = reinterpret_cast<uint32_t const *>(code.data()),
		};
		handle = device_->createShaderModule(&createInfo);
	}
};

struct VulkanGraphicsPipeline : public VulkanHandle<VkPipeline> {
protected:
	//owned:
	VkPipelineLayout pipelineLayout = {};
	std::unique_ptr<VulkanDescriptorSetLayout> descriptorSetLayout;
	
	//held:
	VulkanDevice const * const device = {};				//held for dtor
public:
	VkPipelineLayout getPipelineLayout() const { return ASSERTHANDLE(pipelineLayout); }
	
	VulkanDescriptorSetLayout * getDescriptorSetLayout() { return descriptorSetLayout.get(); }
	VulkanDescriptorSetLayout const * getDescriptorSetLayout() const { return descriptorSetLayout.get(); }

	~VulkanGraphicsPipeline() {
		if (pipelineLayout) vkDestroyPipelineLayout((*device)(), pipelineLayout, getAllocator());
		if (handle) vkDestroyPipeline((*device)(), handle, getAllocator());
		descriptorSetLayout = nullptr;
	}

	VulkanGraphicsPipeline(
		VulkanPhysicalDevice const * const physicalDevice,
		VulkanDevice const * const device_,
		VulkanRenderPass const * const renderPass
	) : device(device_) {
		
		// descriptorSetLayout is only used by graphicsPipeline
		descriptorSetLayout = std::make_unique<VulkanDescriptorSetLayout>(device_);

		auto vertShaderModule = VulkanShaderModule(
			device_,
			Common::File::read("shader-vert.spv")
		);
		auto vertShaderStageInfo = VkPipelineShaderStageCreateInfo{
			.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
			.stage = VK_SHADER_STAGE_VERTEX_BIT,
			.module = vertShaderModule(),
			.pName = "main",
			//.pName = "vert",		// GLSL uses 'main', but clspv doesn't allow 'main', so ....
		};
		
		auto fragShaderModule = VulkanShaderModule(
			device_,
			Common::File::read("shader-frag.spv")
		);
		auto fragShaderStageInfo = VkPipelineShaderStageCreateInfo{
			.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
			.stage = VK_SHADER_STAGE_FRAGMENT_BIT,
			.module = fragShaderModule(),
			.pName = "main",
			//.pName = "frag",
		};
		
		auto bindingDescriptions = Common::make_array(
			Vertex::getBindingDescription()
		);
		auto attributeDescriptions = Vertex::getAttributeDescriptions();
		auto vertexInputInfo = VkPipelineVertexInputStateCreateInfo{
			.sType = VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO,
			.vertexBindingDescriptionCount = (uint32_t)bindingDescriptions.size(),
			.pVertexBindingDescriptions = bindingDescriptions.data(),
			.vertexAttributeDescriptionCount = (uint32_t)attributeDescriptions.size(),
			.pVertexAttributeDescriptions = attributeDescriptions.data(),
		};

		auto inputAssembly = VkPipelineInputAssemblyStateCreateInfo{
			.sType = VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO,
			.topology = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST,
			.primitiveRestartEnable = VK_FALSE,
		};

		auto viewportState = VkPipelineViewportStateCreateInfo{
			.sType = VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO,
			.viewportCount = 1,
			.scissorCount = 1,
		};

		auto rasterizer = VkPipelineRasterizationStateCreateInfo{
			.sType = VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO,
			.depthClampEnable = VK_FALSE,
			.rasterizerDiscardEnable = VK_FALSE,
			.polygonMode = VK_POLYGON_MODE_FILL,
			//.cullMode = VK_CULL_MODE_BACK_BIT,
			//.frontFace = VK_FRONT_FACE_CLOCKWISE,
			//.frontFace = VK_FRONT_FACE_COUNTER_CLOCKWISE,
			.depthBiasEnable = VK_FALSE,
			.lineWidth = 1,
		};

		auto multisampling = VkPipelineMultisampleStateCreateInfo{
			.sType = VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO,
			.rasterizationSamples = physicalDevice->getMSAASamples(),
			.sampleShadingEnable = VK_FALSE,
		};

        auto depthStencil = VkPipelineDepthStencilStateCreateInfo{
			.sType = VK_STRUCTURE_TYPE_PIPELINE_DEPTH_STENCIL_STATE_CREATE_INFO,
			.depthTestEnable = VK_TRUE,
			.depthWriteEnable = VK_TRUE,
			.depthCompareOp = VK_COMPARE_OP_LESS,
			.depthBoundsTestEnable = VK_FALSE,
			.stencilTestEnable = VK_FALSE,
		};

		auto colorBlendAttachment = VkPipelineColorBlendAttachmentState{
			.blendEnable = VK_FALSE,
			.colorWriteMask = VK_COLOR_COMPONENT_R_BIT | VK_COLOR_COMPONENT_G_BIT | VK_COLOR_COMPONENT_B_BIT | VK_COLOR_COMPONENT_A_BIT,
		};

		auto colorBlending = VkPipelineColorBlendStateCreateInfo{
			.sType = VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO,
			.logicOpEnable = VK_FALSE,
			.logicOp = VK_LOGIC_OP_COPY,
			.attachmentCount = 1,
			.pAttachments = &colorBlendAttachment,
			.blendConstants = {0,0,0,0},
		};

		auto dynamicStates = Common::make_array<VkDynamicState>(
			VK_DYNAMIC_STATE_VIEWPORT,
			VK_DYNAMIC_STATE_SCISSOR
		);
		auto dynamicState = VkPipelineDynamicStateCreateInfo{
			.sType = VK_STRUCTURE_TYPE_PIPELINE_DYNAMIC_STATE_CREATE_INFO,
			.dynamicStateCount = (uint32_t)dynamicStates.size(),
			.pDynamicStates = dynamicStates.data(),
		};
		
		auto descriptorSetLayouts = Common::make_array<VkDescriptorSetLayout>(
			(*descriptorSetLayout)()
		);
		auto pipelineLayoutInfo = VkPipelineLayoutCreateInfo{
			.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO,
			.setLayoutCount = (uint32_t)descriptorSetLayouts.size(),
			.pSetLayouts = descriptorSetLayouts.data(),
		};
		pipelineLayout = device_->createPipelineLayout(&pipelineLayoutInfo);

		auto shaderStages = Common::make_array(
			vertShaderStageInfo,
			fragShaderStageInfo
		);
		auto pipelineInfo = VkGraphicsPipelineCreateInfo{
			.sType = VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO,
			.stageCount = (uint32_t)shaderStages.size(),
			.pStages = shaderStages.data(),
			.pVertexInputState = &vertexInputInfo,
			.pInputAssemblyState = &inputAssembly,
			.pViewportState = &viewportState,
			.pRasterizationState = &rasterizer,
			.pMultisampleState = &multisampling,
			.pDepthStencilState = &depthStencil,
			.pColorBlendState = &colorBlending,
			.pDynamicState = &dynamicState,
			.layout = pipelineLayout,
			.renderPass = (*renderPass)(),
			.subpass = 0,
			.basePipelineHandle = VK_NULL_HANDLE,
		};
		handle = device_->createGraphicsPipelines((VkPipelineCache)VK_NULL_HANDLE, 1, &pipelineInfo, getAllocator());
	}
};

// so I don't have to prefix all my fields and names
struct VulkanCommon {
protected:
	static constexpr std::string modelPath = "viking_room.obj";
	static constexpr std::string texturePath = "viking_room.png";
	static constexpr int maxFramesInFlight = 2;

	::SDLApp::SDLApp const * app = {};	// points back to the owner

#if 0	// not working on my vulkan implementation
	static constexpr bool const enableValidationLayers = true;
#else
	static constexpr bool const enableValidationLayers = false;
#endif

	std::unique_ptr<VulkanInstance> instance;
	std::unique_ptr<VulkanDebugMessenger> debug;	// optional
	std::unique_ptr<VulkanSurface> surface;
	std::unique_ptr<VulkanDevice> device;
	std::unique_ptr<VulkanSwapChain> swapChain;
	std::unique_ptr<VulkanGraphicsPipeline> graphicsPipeline;
	std::unique_ptr<VulkanCommandPool> commandPool;
	std::unique_ptr<VulkanDeviceMemoryBuffer> vertexBuffer;
	std::unique_ptr<VulkanDeviceMemoryBuffer> indexBuffer;
	
	uint32_t mipLevels = {};
	std::unique_ptr<VulkanDeviceMemoryImage> textureImage;

	std::unique_ptr<VulkanImageView> textureImageView;
	std::unique_ptr<VulkanSampler> textureSampler;
    
	std::vector<Vertex> vertices;
    std::vector<uint32_t> indices;

	// hmm combine these two into a class?
	std::vector<std::unique_ptr<VulkanDeviceMemoryBuffer>> uniformBuffers;
	std::vector<void*> uniformBuffersMapped;
	
	std::unique_ptr<VulkanDescriptorPool> descriptorPool;
	
	// each of these, there are one per number of frames in flight
	std::vector<VkDescriptorSet> descriptorSets;
	std::vector<VkCommandBuffer> commandBuffers;
	std::vector<VkSemaphore> imageAvailableSemaphores;
	std::vector<VkSemaphore> renderFinishedSemaphores;
	std::vector<VkFence> inFlightFences;
	
	uint32_t currentFrame = {};
	
	bool framebufferResized = {};
public:
	void setFramebufferResized() { framebufferResized = true; }
protected:

	// used by
	//	VulkanPhysicalDevice::checkDeviceExtensionSupport
	//	initLogicalDevice
	std::vector<char const *> const deviceExtensions = {
		VK_KHR_SWAPCHAIN_EXTENSION_NAME
	};

	//ok now we're at the point where we are recreating objects dependent on physicalDevice so
	std::unique_ptr<VulkanPhysicalDevice> physicalDevice;
public:
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
		
		surface = std::make_unique<VulkanSurface>(
			app->getWindow(),
			instance.get()
		);
		physicalDevice = std::make_unique<VulkanPhysicalDevice>(
			instance.get(),
			(*surface)(),
			deviceExtensions
		);
		device = std::make_unique<VulkanDevice>(
			physicalDevice.get(),
			surface.get(),
			deviceExtensions,
			enableValidationLayers
		);
		swapChain = std::make_unique<VulkanSwapChain>(
			app->getScreenSize(),
			physicalDevice.get(),
			device.get(),
			surface.get()
		);
		graphicsPipeline = std::make_unique<VulkanGraphicsPipeline>(
			physicalDevice.get(),
			device.get(),
			swapChain->getRenderPass()
		);
		commandPool = std::make_unique<VulkanCommandPool>(
			physicalDevice.get(),
			device.get(),
			(*surface)()
		);
	
		createTextureImage();
       
		textureImageView = swapChain->createImageView(
			(*textureImage)(),
			VK_FORMAT_R8G8B8A8_SRGB,
			VK_IMAGE_ASPECT_COLOR_BIT,
			mipLevels
		);

		textureSampler = std::make_unique<VulkanSampler>(
			device.get(),
			VkSamplerCreateInfo{
				.sType = VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO,
				.magFilter = VK_FILTER_LINEAR,
				.minFilter = VK_FILTER_LINEAR,
				.mipmapMode = VK_SAMPLER_MIPMAP_MODE_LINEAR,
				.addressModeU = VK_SAMPLER_ADDRESS_MODE_REPEAT,
				.addressModeV = VK_SAMPLER_ADDRESS_MODE_REPEAT,
				.addressModeW = VK_SAMPLER_ADDRESS_MODE_REPEAT,
				.mipLodBias = 0,
				.anisotropyEnable = VK_TRUE,
				.maxAnisotropy = physicalDevice->getProperties().limits.maxSamplerAnisotropy,
				.compareEnable = VK_FALSE,
				.compareOp = VK_COMPARE_OP_ALWAYS,
				.minLod = 0,
				.maxLod = static_cast<float>(mipLevels),
				.borderColor = VK_BORDER_COLOR_INT_OPAQUE_BLACK,
				.unnormalizedCoordinates = VK_FALSE,
			}
		);

		loadModel();
		
		initVertexBuffer();
		initIndexBuffer();
		initUniformBuffers();
		
		descriptorPool = std::make_unique<VulkanDescriptorPool>(
			device.get(),
			(uint32_t)maxFramesInFlight
		);
		
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

public:
	~VulkanCommon() {
		swapChain = nullptr;
		graphicsPipeline = nullptr;
		uniformBuffers.clear();
		indexBuffer = nullptr;
		vertexBuffer = nullptr;
		descriptorPool = nullptr;		
		textureSampler = nullptr;
		textureImageView = nullptr;
		textureImage = nullptr;

		for (size_t i = 0; i < maxFramesInFlight; i++) {
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
		std::shared_ptr<Image::Image> image = std::dynamic_pointer_cast<Image::Image>(Image::system->read(texturePath));
		if (!image) {
			throw Common::Exception() << "failed to load image from " << texturePath;
		}
		auto texSize = image->getSize();
		
		// TODO move this into Image::Image setBitsPerPixel() or something
		int texBPP = image->getBitsPerPixel() >> 3;
		constexpr int desiredBPP = 4;
		if (texBPP != desiredBPP) {
			//resample
			auto newimage = std::make_shared<Image::Image>(image->getSize(), nullptr, desiredBPP);
			for (int i = 0; i < texSize.x * texSize.y; ++i) {
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
		
		char const * const srcData = image->getData();
		VkDeviceSize const bufferSize = texSize.x * texSize.y * texBPP;
		mipLevels = (uint32_t)std::floor(std::log2(std::max(texSize.x, texSize.y))) + 1;
	
		textureImage = VulkanDeviceMemoryImage::makeTextureFromStaged(
			physicalDevice.get(),
			device.get(),
			commandPool.get(),
			srcData,
			bufferSize,
			texSize.x,
			texSize.y,
			mipLevels
		);
	
        generateMipmaps(
			(*textureImage)(),
			VK_FORMAT_R8G8B8A8_SRGB,
			texSize.x,
			texSize.y,
			mipLevels
		);
	}

    void generateMipmaps(
		VkImage image,
		VkFormat imageFormat,
		int32_t texWidth,
		int32_t texHeight,
		uint32_t mipLevels
	) {
        // Check if image format supports linear blitting
        auto formatProperties = physicalDevice->getFormatProperties(imageFormat);

        if (!(formatProperties.optimalTilingFeatures & VK_FORMAT_FEATURE_SAMPLED_IMAGE_FILTER_LINEAR_BIT)) {
            throw Common::Exception() << "texture image format does not support linear blitting!";
        }

        VulkanSingleTimeCommand commandBuffer(device.get(), (*commandPool)());

        auto barrier = VkImageMemoryBarrier{
			.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER,
			.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
			.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
			.image = image,
			.subresourceRange = {
				.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT,
				.levelCount = 1,
				.baseArrayLayer = 0,
				.layerCount = 1,
			},
		};
        
		int32_t mipWidth = texWidth;
        int32_t mipHeight = texHeight;

        for (uint32_t i = 1; i < mipLevels; i++) {
            barrier.subresourceRange.baseMipLevel = i - 1;
            barrier.oldLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
            barrier.newLayout = VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL;
            barrier.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
            barrier.dstAccessMask = VK_ACCESS_TRANSFER_READ_BIT;

            vkCmdPipelineBarrier(
				commandBuffer(),
                VK_PIPELINE_STAGE_TRANSFER_BIT,
				VK_PIPELINE_STAGE_TRANSFER_BIT,
				0,
                0,
				nullptr,
                0,
				nullptr,
                1,
				&barrier
			);

            auto blit = VkImageBlit{
				.srcSubresource = {
					.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT,
					.mipLevel = i - 1,
					.baseArrayLayer = 0,
					.layerCount = 1,
				},
				.srcOffsets = {
					{0, 0, 0},
					{mipWidth, mipHeight, 1},
				},
				.dstSubresource = {
					.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT,
					.mipLevel = i,
					.baseArrayLayer = 0,
					.layerCount = 1,
				},
				.dstOffsets = {
					{0, 0, 0},
					{
						mipWidth > 1 ? mipWidth / 2 : 1,
						mipHeight > 1 ? mipHeight / 2 : 1,
						1,
					},
				},
			};
			vkCmdBlitImage(
				commandBuffer(),
                image,
				VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
                image,
				VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
                1,
				&blit,
                VK_FILTER_LINEAR);

            barrier.oldLayout = VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL;
            barrier.newLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
            barrier.srcAccessMask = VK_ACCESS_TRANSFER_READ_BIT;
            barrier.dstAccessMask = VK_ACCESS_SHADER_READ_BIT;

            vkCmdPipelineBarrier(
				commandBuffer(),
                VK_PIPELINE_STAGE_TRANSFER_BIT,
				VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT,
				0,
                0,
				nullptr,
                0,
				nullptr,
                1,
				&barrier
			);

            if (mipWidth > 1) mipWidth /= 2;
            if (mipHeight > 1) mipHeight /= 2;
        }

        barrier.subresourceRange.baseMipLevel = mipLevels - 1;
        barrier.oldLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
        barrier.newLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
        barrier.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
        barrier.dstAccessMask = VK_ACCESS_SHADER_READ_BIT;

        vkCmdPipelineBarrier(
			commandBuffer(),
            VK_PIPELINE_STAGE_TRANSFER_BIT,
			VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT,
			0,
            0,
			nullptr,
            0,
			nullptr,
            1,
			&barrier
		);
    }

    void loadModel() {
        tinyobj::attrib_t attrib;
        std::vector<tinyobj::shape_t> shapes;
        std::vector<tinyobj::material_t> materials;
        std::string warn, err;
        if (!tinyobj::LoadObj(&attrib, &shapes, &materials, &warn, &err, modelPath.c_str())) {
            throw Common::Exception() << warn << err;
        }

        std::unordered_map<Vertex, uint32_t> uniqueVertices;

        for (const auto& shape : shapes) {
            for (const auto& index : shape.mesh.indices) {
                Vertex vertex;

                vertex.pos = {
                    attrib.vertices[3 * index.vertex_index + 0],
                    attrib.vertices[3 * index.vertex_index + 1],
                    attrib.vertices[3 * index.vertex_index + 2]
                };

                vertex.texCoord = {
                    attrib.texcoords[2 * index.texcoord_index + 0],
                    1.0f - attrib.texcoords[2 * index.texcoord_index + 1]
                };

                vertex.color = {1.0f, 1.0f, 1.0f};

                if (uniqueVertices.count(vertex) == 0) {
                    uniqueVertices[vertex] = (uint32_t)vertices.size();
                    vertices.push_back(vertex);
                }

                indices.push_back(uniqueVertices[vertex]);
            }
        }
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
			surface.get()
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

		uniformBuffers.resize(maxFramesInFlight);
		uniformBuffersMapped.resize(maxFramesInFlight);

		for (size_t i = 0; i < maxFramesInFlight; i++) {
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
		commandBuffers.resize(maxFramesInFlight);
		// TODO this matches 'VulkanSingleTimeCommand' ctor
		auto allocInfo = VkCommandBufferAllocateInfo{
			.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO,
			.commandPool = (*commandPool)(),
			.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY,
			.commandBufferCount = (uint32_t)commandBuffers.size(),
		};
		VULKAN_SAFE(vkAllocateCommandBuffers, (*device)(), &allocInfo, commandBuffers.data());
		// end part that matches
	}

	void recordCommandBuffer(VkCommandBuffer commandBuffer, uint32_t imageIndex) {
		// TODO this part matches VulkanSingleTimeCommand ctor
		auto beginInfo = VkCommandBufferBeginInfo{
			.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO,
		};
		VULKAN_SAFE(vkBeginCommandBuffer, commandBuffer, &beginInfo);
		// end part that matches

		auto clearValues = Common::make_array(
			VkClearValue{
				.color = {{0, 0, 0, 1}},
			},
			VkClearValue{
				.depthStencil = {1, 0},
			}
		);
		auto renderPassInfo = VkRenderPassBeginInfo{
			.sType = VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO,
			.renderPass = (*swapChain->getRenderPass())(),
			.framebuffer = swapChain->framebuffers[imageIndex],
			.renderArea = {
				.offset = {0, 0},
				.extent = swapChain->extent,
			},
			.clearValueCount = (uint32_t)clearValues.size(),
			.pClearValues = clearValues.data(),
		};
		vkCmdBeginRenderPass(commandBuffer, &renderPassInfo, VK_SUBPASS_CONTENTS_INLINE);

		{
			vkCmdBindPipeline(
				commandBuffer,
				VK_PIPELINE_BIND_POINT_GRAPHICS,
				(*graphicsPipeline)()
			);

			auto viewport = VkViewport{
				.x = 0,
				.y = 0,
				.width = (float)swapChain->extent.width,
				.height = (float)swapChain->extent.height,
				.minDepth = 0,
				.maxDepth = 1,
			};
			vkCmdSetViewport(commandBuffer, 0, 1, &viewport);

			auto scissor = VkRect2D{
				.offset = {0, 0},
				.extent = swapChain->extent,
			};
			vkCmdSetScissor(commandBuffer, 0, 1, &scissor);

			auto vertexBuffers = Common::make_array<VkBuffer>(
				(*vertexBuffer)()
			);
			VkDeviceSize offsets[] = {0};
			vkCmdBindVertexBuffers(
				commandBuffer,
				0,
				(uint32_t)vertexBuffers.size(),
				vertexBuffers.data(),
				offsets
			);

			vkCmdBindIndexBuffer(
				commandBuffer,
				(*indexBuffer)(),
				0,
				VK_INDEX_TYPE_UINT32
			);

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

			vkCmdDrawIndexed(
				commandBuffer,
				(uint32_t)indices.size(),
				1,
				0,
				0,
				0
			);
		}

		vkCmdEndRenderPass(commandBuffer);

		VULKAN_SAFE(vkEndCommandBuffer, commandBuffer);
	}

	void initSyncObjects() {
		imageAvailableSemaphores.resize(maxFramesInFlight);
		renderFinishedSemaphores.resize(maxFramesInFlight);
		inFlightFences.resize(maxFramesInFlight);

		auto semaphoreInfo = VkSemaphoreCreateInfo{
			.sType = VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO,
		};

		auto fenceInfo = VkFenceCreateInfo{
			.sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO,
			.flags = VK_FENCE_CREATE_SIGNALED_BIT,
		};

		for (size_t i = 0; i < maxFramesInFlight; i++) {
			VULKAN_SAFE(vkCreateSemaphore, (*device)(), &semaphoreInfo, nullptr, &imageAvailableSemaphores[i]);
			VULKAN_SAFE(vkCreateSemaphore, (*device)(), &semaphoreInfo, nullptr, &renderFinishedSemaphores[i]);
			VULKAN_SAFE(vkCreateFence, (*device)(), &fenceInfo, nullptr, &inFlightFences[i]);
		}
	}

	void createDescriptorSets() {
		std::vector<VkDescriptorSetLayout> layouts(maxFramesInFlight, (*graphicsPipeline->getDescriptorSetLayout())());
		auto allocInfo = VkDescriptorSetAllocateInfo{
			.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO,
			.descriptorPool = (*descriptorPool)(),
			.descriptorSetCount = (uint32_t)maxFramesInFlight,
			.pSetLayouts = layouts.data(),
		};
		descriptorSets.resize(maxFramesInFlight);
		VULKAN_SAFE(vkAllocateDescriptorSets, (*device)(), &allocInfo, descriptorSets.data());

		for (size_t i = 0; i < maxFramesInFlight; i++) {
			auto bufferInfo = VkDescriptorBufferInfo{
				.buffer = (*uniformBuffers[i])(),
				.offset = 0,
				.range = sizeof(UniformBufferObject),
			};
			auto imageInfo = VkDescriptorImageInfo{
				.sampler = (*textureSampler)(),
				.imageView = (*textureImageView)(),
				.imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
			};
			auto descriptorWrites = Common::make_array(
				VkWriteDescriptorSet{
					.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
					.dstSet = descriptorSets[i],
					.dstBinding = 0,
					.descriptorCount = 1,
					.descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER,
					.pBufferInfo = &bufferInfo,
				},
				VkWriteDescriptorSet{
					.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
					.dstSet = descriptorSets[i],
					.dstBinding = 1,
					.descriptorCount = 1,
					.descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
					.pImageInfo = &imageInfo,
				}
			);
			vkUpdateDescriptorSets(
				(*device)(),
				(uint32_t)descriptorWrites.size(),
				descriptorWrites.data(),
				0,
				nullptr
			);
		}
	}

	decltype(std::chrono::high_resolution_clock::now()) startTime = std::chrono::high_resolution_clock::now();
	
	void updateUniformBuffer(uint32_t currentFrame_) {
		//static auto startTime = std::chrono::high_resolution_clock::now();
		auto currentTime = std::chrono::high_resolution_clock::now();
		float time = std::chrono::duration<float, std::chrono::seconds::period>(currentTime - startTime).count();

		auto ubo = UniformBufferObject{};
		ubo.model = Tensor::rotate<float>(
			Tensor::float4i4(1),
			time * degToRad<float>(90),
			Tensor::float3(0, 0, 1)
		);
		//isn't working ...
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
/*
working buffer.  in-order in memory as it gets passed to Vulkan:
float[3][4][4] buf = {
	//model
	{
		{-0.724425, 0.689354, 0.000000, 0.000000},
		{-0.689354, -0.724425, 0.000000, 0.000000},
		{0.000000, 0.000000, 1.000000, 0.000000},
		{0.000000, 0.000000, 0.000000, 1.000000},
	},
	//view
	{
		{-0.707107, -0.408248, 0.577350, 0.000000},
		{0.707107, -0.408248, 0.577350, 0.000000},
		{0.000000, 0.816497, 0.577350, 0.000000},
		{-0.000000, -0.000000, -3.464102, 1.000000},
	},
	//proj
	{
		{1.810660, 0.000000, 0.000000, 0.000000},
		{0.000000, -2.414213, 0.000000, 0.000000},
		{0.000000, 0.000000, -1.020202, -1.000000},
		{0.000000, 0.000000, -0.202020, 0.000000},
	},
};
*/
		// I use row-major, Vulkan/GL uses column-major
		ubo.model = ubo.model.transpose();
		ubo.view = ubo.view.transpose();
		ubo.proj = ubo.proj.transpose();

		ubo.view = {
			{-0.707107, -0.408248, 0.577350, 0.000000},
			{0.707107, -0.408248, 0.577350, 0.000000},
			{0.000000, 0.816497, 0.577350, 0.000000},
			{-0.000000, -0.000000, -3.464102, 1.000000},
		};

//std::cout << ubo.view << std::endl;	
/*
{
	{0.707107, -0.707107, 0, 0},
	{-0.408248, -0.408248, 0.816497, 0},
	{0.57735, 0.57735, 0.57735, 0},
	{-0, -0, 3.4641, 1}
}
*/
		memcpy(uniformBuffersMapped[currentFrame_], &ubo, sizeof(ubo));
	}
public:
	void drawFrame() {
		vkWaitForFences((*device)(), 1, &inFlightFences[currentFrame], VK_TRUE, UINT64_MAX);

		auto imageIndex = uint32_t{};
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

		auto waitSemaphores = Common::make_array(
			(VkSemaphore)imageAvailableSemaphores[currentFrame]
		);
		auto waitStages = Common::make_array<VkPipelineStageFlags>(
			VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT
		);
		static_assert(waitSemaphores.size() == waitStages.size());
		
		auto signalSemaphores = Common::make_array(
			(VkSemaphore)renderFinishedSemaphores[currentFrame]
		);

		// static assert sizes match?
		auto submitInfo = VkSubmitInfo{
			.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO,
			.waitSemaphoreCount = (uint32_t)waitSemaphores.size(),
			.pWaitSemaphores = waitSemaphores.data(),
			.pWaitDstStageMask = waitStages.data(),
			.commandBufferCount = 1,
			.pCommandBuffers = &commandBuffers[currentFrame],
			.signalSemaphoreCount = (uint32_t)signalSemaphores.size(),
			.pSignalSemaphores = signalSemaphores.data(),
		};
		device->getGraphicsQueue()->submit(1, &submitInfo, inFlightFences[currentFrame]);
		
		auto swapChains = Common::make_array<VkSwapchainKHR>(
			(*swapChain)()
		);
		auto presentInfo = VkPresentInfoKHR{
			.sType = VK_STRUCTURE_TYPE_PRESENT_INFO_KHR,
			.waitSemaphoreCount = (uint32_t)signalSemaphores.size(),
			// these two sizes need t match (right?)
			.pWaitSemaphores = signalSemaphores.data(),
			.swapchainCount = (uint32_t)swapChains.size(),
			// wait do these two sizes need to match?
			.pSwapchains = swapChains.data(),
			.pImageIndices = &imageIndex,
		};
		result = device->getPresentQueue()->present(&presentInfo);
		if (result == VK_ERROR_OUT_OF_DATE_KHR || result == VK_SUBOPTIMAL_KHR || framebufferResized) {
			framebufferResized = false;
			recreateSwapChain();
		} else if (result != VK_SUCCESS) {
			throw Common::Exception() << "vkQueuePresentKHR failed: " << result;
		}

		currentFrame = (currentFrame + 1) % maxFramesInFlight;
	}
public:
	void loopDone() {
		device->waitIdle();
	}
};

struct Test : public ::SDLApp::SDLApp {
	using Super = ::SDLApp::SDLApp;

protected:
	std::unique_ptr<VulkanCommon> vk;
	
	virtual void initWindow() {
		Super::initWindow();
		vk = std::make_unique<VulkanCommon>(this);
	}

	virtual std::string getTitle() const {
		return "Vulkan Test";
	}
	
	virtual Uint32 getSDLCreateWindowFlags() {
		auto flags = Super::getSDLCreateWindowFlags();
		flags |= SDL_WINDOW_VULKAN;
//		flags &= ~SDL_WINDOW_RESIZABLE;
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
		vk->setFramebufferResized();
	}
};

SDLAPP_MAIN(Test)
