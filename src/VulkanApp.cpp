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
#include <vulkan/vulkan_raii.hpp>
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
	if (res != (VkResult)vk::Result::eSuccess) {
		throw Common::Exception() << what << " failed: " << res;
	}
}

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

auto & assertHandle(auto & x, char const * where) {
	if (!x) throw Common::Exception() << "returned an empty handle at " << where;
	return x;
}
auto const & assertHandle(auto const & x, char const * where) {
	if (!x) throw Common::Exception() << "returned an empty handle at " << where;
	return x;
}
auto && assertHandle(auto && x, char const * where) {
	if (!x) throw Common::Exception() << "returned an empty handle at " << where;
	return std::move(x);
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
		return vk::VertexInputBindingDescription(VkVertexInputBindingDescription{
			.binding = 0,
			.stride = sizeof(Vertex),
			.inputRate = (VkVertexInputRate)vk::VertexInputRate::eVertex,
		});
	}

	static auto getAttributeDescriptions() {
		return Common::make_array(
			VkVertexInputAttributeDescription{
				.location = 0,
				.binding = 0,
				.format = (VkFormat)vk::Format::eR32G32B32Sfloat,
				.offset = offsetof(Vertex, pos),
			},
			VkVertexInputAttributeDescription{
				.location = 1,
				.binding = 0,
				.format = (VkFormat)vk::Format::eR32G32B32Sfloat,
				.offset = offsetof(Vertex, color),
			},
			VkVertexInputAttributeDescription{
				.location = 2,
				.binding = 0,
				.format = (VkFormat)vk::Format::eR32G32Sfloat,
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
	__attribute__ ((packed)) Tensor::float4x4 model;
	__attribute__ ((packed)) Tensor::float4x4 view;
	__attribute__ ((packed)) Tensor::float4x4 proj;
};
static_assert(sizeof(UniformBufferObject) == 4 * 4 * sizeof(float) * 3);

struct ThisVulkanDebugMessenger {
	static auto create(
		vk::Instance const & instance
	) {
#if 0 // how do I compile this?		
		obj = vk::DebugUtilsMessengerEXT(
			instance,
			vk::DebugUtilsMessengerCreateInfoEXT(
				{},
				vk::DebugUtilsMessageSeverityFlagBitsEXT::eVerbose 
					| vk::DebugUtilsMessageSeverityFlagBitsEXT::eWarning
					| vk::DebugUtilsMessageSeverityFlagBitsEXT::eError,
				vk::DebugUtilsMessageTypeFlagBitsEXT::eGeneral
					| vk::DebugUtilsMessageTypeFlagBitsEXT::eValidation
					| vk::DebugUtilsMessageTypeFlagBitsEXT::ePerformance,
				debugCallback
			)	
		);
#endif
		return vk::DebugUtilsMessengerEXT();
	}

// app-specific callback
protected:
	static VKAPI_ATTR VkBool32 VKAPI_CALL debugCallback(VkDebugUtilsMessageSeverityFlagBitsEXT messageSeverity, VkDebugUtilsMessageTypeFlagsEXT messageType, const VkDebugUtilsMessengerCallbackDataEXT* pCallbackData, void* pUserData) {
		std::cerr << "validation layer: " << pCallbackData->pMessage << std::endl;
		return VK_FALSE;
	}
};


// ************** from here on down, app-specific **************


struct ThisVulkanInstance {

	// this does result in vkCreateInstance,
	//  but the way it gest there is very application-specific
	static auto create(
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
			.sType = (VkStructureType)vk::StructureType::eApplicationInfo,
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

		// would be nice to do a parent-constructor-call here
		//  but I can't move this into a static method that passes it into super
		//  because the result uses pointers to other stack/temp objects
		auto createInfo = VkInstanceCreateInfo{
			.sType = (VkStructureType)vk::StructureType::eInstanceCreateInfo,
			.pApplicationInfo = &appInfo,
			.enabledLayerCount = (uint32_t)layerNames.size(),
			.ppEnabledLayerNames = layerNames.data(),
			.enabledExtensionCount = (uint32_t)extensions.size(),
			.ppEnabledExtensionNames = extensions.data(),
		};
		
		return vk::createInstance(createInfo);
	}

protected:
	static std::vector<char const *> getRequiredExtensions(
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

struct ThisVulkanPhysicalDevice {
	// used by the application for specific physical device querying (should be a subclass of the general vk::PhysicalDevice)
	static auto create(
		vk::Instance const & instance,
		vk::SurfaceKHR surface,
		std::vector<char const *> const & deviceExtensions
	) {
		vk::PhysicalDevice desiredPhysDev = {};
		auto physDevs = instance.enumeratePhysicalDevices();
		//debug:
		std::cout << "devices:" << std::endl;
		for (auto const & physDev : physDevs) {
			auto props = physDev.getProperties();
			std::cout
				<< "\t"
				<< std::string(props.deviceName)
				<< " type=" << (uint32_t)props.deviceType
				<< std::endl;
		}

		for (auto const & physDev : physDevs) {
			if (isDeviceSuitable(physDev, surface, deviceExtensions)) {
				desiredPhysDev = physDev;
				break;
			}
		}

		if (!desiredPhysDev) throw Common::Exception() << "failed to find a suitable GPU!";
		return desiredPhysDev;
	}

public:
	struct SwapChainSupportDetails {
		vk::SurfaceCapabilitiesKHR capabilities;
		std::vector<vk::SurfaceFormatKHR> formats;
		std::vector<vk::PresentModeKHR> presentModes;
	};

	static auto querySwapChainSupport(
		vk::PhysicalDevice physDev,
		vk::SurfaceKHR surface
	) {
		return SwapChainSupportDetails{
			.capabilities = physDev.getSurfaceCapabilitiesKHR(surface),
			.formats = physDev.getSurfaceFormatsKHR(surface),
			.presentModes = physDev.getSurfacePresentModesKHR(surface)
		};
	}

protected:
	static bool isDeviceSuitable(
		vk::PhysicalDevice physDev,
		vk::SurfaceKHR surface,
		std::vector<char const *> const & deviceExtensions
	) {
		auto indices = findQueueFamilies(physDev, surface);
		bool extensionsSupported = checkDeviceExtensionSupport(physDev, deviceExtensions);
		bool swapChainAdequate = false;
		if (extensionsSupported) {
			auto swapChainSupport = querySwapChainSupport(physDev, surface);
			swapChainAdequate = !swapChainSupport.formats.empty() && !swapChainSupport.presentModes.empty();
		}
		VkPhysicalDeviceFeatures features = physDev.getFeatures();
		return indices.isComplete()
			&& extensionsSupported
			&& swapChainAdequate
			&& features.samplerAnisotropy;
	}

	static bool checkDeviceExtensionSupport(
		vk::PhysicalDevice physDev,
		std::vector<char const *> const & deviceExtensions
	) {
		std::set<std::string> requiredExtensions(deviceExtensions.begin(), deviceExtensions.end());
		for (auto const & extension : physDev.enumerateDeviceExtensionProperties()) {
			requiredExtensions.erase(extension.extensionName);
		}
		return requiredExtensions.empty();
	}

public:
	static vk::SampleCountFlagBits getMaxUsableSampleCount(
		vk::PhysicalDevice physDev
	) {
		auto props = physDev.getProperties();
		auto counts = props.limits.framebufferColorSampleCounts 
			& props.limits.framebufferDepthSampleCounts;
		if (counts & vk::SampleCountFlagBits::e64) { return vk::SampleCountFlagBits::e64; }
		if (counts & vk::SampleCountFlagBits::e32) { return vk::SampleCountFlagBits::e32; }
		if (counts & vk::SampleCountFlagBits::e16) { return vk::SampleCountFlagBits::e16; }
		if (counts & vk::SampleCountFlagBits::e8) { return vk::SampleCountFlagBits::e8; }
		if (counts & vk::SampleCountFlagBits::e4) { return vk::SampleCountFlagBits::e4; }
		if (counts & vk::SampleCountFlagBits::e2) { return vk::SampleCountFlagBits::e2; }
		return vk::SampleCountFlagBits::e1;
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
	static QueueFamilyIndices findQueueFamilies(
		vk::PhysicalDevice physDev,
		vk::SurfaceKHR surface
	) {
		QueueFamilyIndices indices;
		auto queueFamilies = physDev.getQueueFamilyProperties();
		for (uint32_t i = 0; i < queueFamilies.size(); ++i) {
			auto const & f = queueFamilies[i];
			if (f.queueFlags & vk::QueueFlagBits::eGraphics) {
				indices.graphicsFamily = i;
			}
			if (physDev.getSurfaceSupportKHR(i, surface)) {
				indices.presentFamily = i;
			}
			if (indices.isComplete()) {
				return indices;
			}
		}
		throw Common::Exception() << "couldn't find all indices";
	}

public:
	static auto findDepthFormat(
		vk::PhysicalDevice physDev
	) {
		return findSupportedFormat(
			physDev,
			std::vector<vk::Format>{
				vk::Format::eD32Sfloat,
				vk::Format::eD32SfloatS8Uint,
				vk::Format::eD24UnormS8Uint
			},
			vk::ImageTiling::eOptimal,
			vk::FormatFeatureFlagBits::eDepthStencilAttachment
		);
	}

protected:
	static vk::Format findSupportedFormat(
		vk::PhysicalDevice physDev,
		std::vector<vk::Format> const & candidates,
		vk::ImageTiling const tiling,
		vk::FormatFeatureFlags const features
	) {
		for (auto format : candidates) {
			auto props = physDev.getFormatProperties(format);
			if (tiling == vk::ImageTiling::eLinear &&
				(props.linearTilingFeatures & features) == features
			) {
				return format;
			} else if (tiling == vk::ImageTiling::eOptimal &&
				(props.optimalTilingFeatures & features) == features
			) {
				return format;
			}
		}
		throw Common::Exception() << "failed to find supported format!";
	}

public:
	static uint32_t findMemoryType(
		vk::PhysicalDevice physDev,
		uint32_t mask,
		vk::MemoryPropertyFlags props
	) {
		auto memProps = physDev.getMemoryProperties();
		for (uint32_t i = 0; i < memProps.memoryTypeCount; i++) {
			if ((mask & (1 << i)) && 
				(memProps.memoryTypes[i].propertyFlags & props)
			) {
				return i;
			}
		}
		throw Common::Exception() << ("failed to find suitable memory type!");
	}

};

// validationLayers matches in checkValidationLayerSupport and initLogicalDevice
std::vector<char const *> validationLayers = {
	"VK_LAYER_KHRONOS_validation"
};

namespace VulkanDevice {
	std::tuple<vk::Device, vk::Queue, vk::Queue>
	create(
		vk::PhysicalDevice const physicalDevice,
		vk::SurfaceKHR const surface,
		std::vector<char const *> const & deviceExtensions,
		bool enableValidationLayers
	) {
		auto indices = ThisVulkanPhysicalDevice::findQueueFamilies(physicalDevice, surface);

		auto queuePriorities = Common::make_array<float>(1);
		std::vector<VkDeviceQueueCreateInfo> queueCreateInfos;
		for (uint32_t queueFamily : std::set<uint32_t>{
			indices.graphicsFamily.value(),
			indices.presentFamily.value(),
		}) {
			queueCreateInfos.push_back(VkDeviceQueueCreateInfo{
				.sType = (VkStructureType)vk::StructureType::eDeviceQueueCreateInfo,
				.queueFamilyIndex = queueFamily,
				.queueCount = (uint32_t)queuePriorities.size(),
				.pQueuePriorities = queuePriorities.data(),
			});
		}

		auto deviceFeatures = VkPhysicalDeviceFeatures{
			.samplerAnisotropy = VK_TRUE,
		};
		auto device = vk::Device(physicalDevice.createDevice(VkDeviceCreateInfo{
			.sType = (VkStructureType)vk::StructureType::eDeviceCreateInfo,
			.queueCreateInfoCount = (uint32_t)queueCreateInfos.size(),
			.pQueueCreateInfos = queueCreateInfos.data(),
			.enabledLayerCount = enableValidationLayers ? (uint32_t)validationLayers.size() : 0,
			.ppEnabledLayerNames = enableValidationLayers ? validationLayers.data() : nullptr,
			.enabledExtensionCount = (uint32_t)deviceExtensions.size(),
			.ppEnabledExtensionNames = deviceExtensions.data(),
			.pEnabledFeatures = &deviceFeatures,
		}));
		auto graphicsQueue = device.getQueue(indices.graphicsFamily.value(), 0);
		auto presentQueue = device.getQueue(indices.presentFamily.value(), 0);
		return std::make_tuple(device, graphicsQueue, presentQueue);
	}
};

struct ThisVulkanRenderPass  {
	static auto create(
		vk::PhysicalDevice const physicalDevice,
		vk::Device const & device,
		vk::Format swapChainImageFormat,
		vk::SampleCountFlagBits msaaSamples
	) {
		auto attachments = Common::make_array(
			VkAttachmentDescription{	//colorAttachment
				.format = (VkFormat)swapChainImageFormat,
				.samples = (VkSampleCountFlagBits)msaaSamples,
				.loadOp = (VkAttachmentLoadOp)vk::AttachmentLoadOp::eClear,
				.storeOp = (VkAttachmentStoreOp)vk::AttachmentStoreOp::eStore,
				.stencilLoadOp = (VkAttachmentLoadOp)vk::AttachmentLoadOp::eDontCare,
				.stencilStoreOp = (VkAttachmentStoreOp)vk::AttachmentStoreOp::eDontCare,
				.initialLayout = (VkImageLayout)vk::ImageLayout::eUndefined,
				.finalLayout = (VkImageLayout)vk::ImageLayout::eColorAttachmentOptimal,
			},
			VkAttachmentDescription{	//depthAttachment
				.format = (VkFormat)ThisVulkanPhysicalDevice::findDepthFormat(physicalDevice),
				.samples = (VkSampleCountFlagBits)msaaSamples,
				.loadOp = (VkAttachmentLoadOp)vk::AttachmentLoadOp::eClear,
				.storeOp = (VkAttachmentStoreOp)vk::AttachmentStoreOp::eDontCare,
				.stencilLoadOp = (VkAttachmentLoadOp)vk::AttachmentLoadOp::eDontCare,
				.stencilStoreOp = (VkAttachmentStoreOp)vk::AttachmentStoreOp::eDontCare,
				.initialLayout = (VkImageLayout)vk::ImageLayout::eUndefined,
				.finalLayout = (VkImageLayout)vk::ImageLayout::eDepthStencilAttachmentOptimal,
			},
			VkAttachmentDescription{	//colorAttachmentResolve
				.format = (VkFormat)swapChainImageFormat,
				.samples = (VkSampleCountFlagBits)vk::SampleCountFlagBits::e1,
				.loadOp = (VkAttachmentLoadOp)vk::AttachmentLoadOp::eDontCare,
				.storeOp = (VkAttachmentStoreOp)vk::AttachmentStoreOp::eStore,
				.stencilLoadOp = (VkAttachmentLoadOp)vk::AttachmentLoadOp::eDontCare,
				.stencilStoreOp = (VkAttachmentStoreOp)vk::AttachmentStoreOp::eDontCare,
				.initialLayout = (VkImageLayout)vk::ImageLayout::eUndefined,
				.finalLayout = (VkImageLayout)vk::ImageLayout::ePresentSrcKHR,
			}
		);
		auto colorAttachmentRef = VkAttachmentReference{
			.attachment = 0,
			.layout = (VkImageLayout)vk::ImageLayout::eColorAttachmentOptimal,
		};
		auto depthAttachmentRef = VkAttachmentReference{
			.attachment = 1,
			.layout = (VkImageLayout)vk::ImageLayout::eDepthStencilAttachmentOptimal,
		};
		auto colorAttachmentResolveRef = VkAttachmentReference{
			.attachment = 2,
			.layout = (VkImageLayout)vk::ImageLayout::eColorAttachmentOptimal,
		};
		auto subpasses = Common::make_array(
			VkSubpassDescription{
				.pipelineBindPoint = (VkPipelineBindPoint)vk::PipelineBindPoint::eGraphics,
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
				.srcStageMask = (VkPipelineStageFlags)(
					vk::PipelineStageFlagBits::eColorAttachmentOutput |
					vk::PipelineStageFlagBits::eEarlyFragmentTests
				),
				.dstStageMask = (VkPipelineStageFlags)(
					vk::PipelineStageFlagBits::eColorAttachmentOutput |
					vk::PipelineStageFlagBits::eEarlyFragmentTests
				),
				.srcAccessMask = 0,
				.dstAccessMask = (VkAccessFlags)(
					vk::AccessFlagBits::eColorAttachmentWrite |
					vk::AccessFlagBits::eDepthStencilAttachmentWrite
				),
			}
		);
		return device.createRenderPass(vk::RenderPassCreateInfo(VkRenderPassCreateInfo{
			.sType = (VkStructureType)vk::StructureType::eRenderPassCreateInfo,
			.attachmentCount = (uint32_t)attachments.size(),
			.pAttachments = attachments.data(),
			.subpassCount = (uint32_t)subpasses.size(),
			.pSubpasses = subpasses.data(),
			.dependencyCount = (uint32_t)dependencies.size(),
			.pDependencies = dependencies.data(),
		}));
	}
};

struct VulkanSingleTimeCommand  {
protected:	
	//owns:
	vk::CommandBuffer cmd;
	//held:
	vk::Device const device;
	vk::Queue const queue;
	vk::CommandPool const commandPool;

public:
	auto const & operator()() const { return cmd; }

	VulkanSingleTimeCommand(
		vk::Device const device_,
		vk::Queue const queue_,
		vk::CommandPool commandPool_
	) : device(device_),
		queue(queue_),
		commandPool(commandPool_)
	{
		cmd = device.allocateCommandBuffers(
			vk::CommandBufferAllocateInfo(
				VkCommandBufferAllocateInfo{
					.sType = (VkStructureType)vk::StructureType::eCommandBufferAllocateInfo,
					.commandPool = commandPool,
					.level = (VkCommandBufferLevel)vk::CommandBufferLevel::ePrimary,
					.commandBufferCount = 1,
				}
			)
		)[0];
		// end part that matches
		// and this part kinda matches the start of 'recordCommandBuffer'
		cmd.begin(vk::CommandBufferBeginInfo(VkCommandBufferBeginInfo{
			.sType = (VkStructureType)vk::StructureType::eCommandBufferBeginInfo,
			.flags = (VkCommandBufferUsageFlags)vk::CommandBufferUsageFlagBits::eOneTimeSubmit,
		}));
		//end part that matches
	}
	
	~VulkanSingleTimeCommand() {
		cmd.end();
		auto cmds = Common::make_array((VkCommandBuffer)cmd);
		queue.submit(vk::SubmitInfo(VkSubmitInfo{
			.sType = (VkStructureType)vk::StructureType::eSubmitInfo,
			.commandBufferCount = (uint32_t)cmds.size(),
			.pCommandBuffers = cmds.data(),
		}));
		queue.waitIdle();
		device.freeCommandBuffers(commandPool, cmd);
	}
};

struct VulkanCommandPool  {
protected:
	//owns:
	vk::CommandPool commandPool;
	//held:
	vk::Device const device;
	vk::Queue const graphicsQueue;
public:
	auto operator()() const { return (VkCommandPool)commandPool; }
	
	~VulkanCommandPool() {
		device.destroyCommandPool(commandPool);
	}
	
	VulkanCommandPool(
		vk::Device const device_,
		vk::Queue const graphicsQueue_,
		vk::CommandPoolCreateInfo const info
	) : commandPool(device_.createCommandPool(info)),
		device(device_),
		graphicsQueue(graphicsQueue_)
	{}

	//copies based on the graphicsQueue
	// used by makeBufferFromStaged
	void copyBuffer(
		vk::Buffer srcBuffer,	//staging VkBuffer
		vk::Buffer dstBuffer,	//dest VkBuffer
		vk::DeviceSize size
	) const {
		VulkanSingleTimeCommand(
			device,
			graphicsQueue,
			(*this)()
		)().copyBuffer(
			srcBuffer,
			dstBuffer,
			vk::BufferCopy(VkBufferCopy{
				.size = size,
			})
		);
	}

	void copyBufferToImage(
		vk::Buffer buffer,
		vk::Image image,
		uint32_t width,
		uint32_t height
	) const {
		VulkanSingleTimeCommand(
			device,
			graphicsQueue,
			(*this)()
		)().copyBufferToImage(
			buffer,
			image,
			vk::ImageLayout::eTransferDstOptimal,
			vk::BufferImageCopy(VkBufferImageCopy{
				.bufferOffset = 0,
				.bufferRowLength = 0,
				.bufferImageHeight = 0,
				.imageSubresource = {
					.aspectMask = (VkImageAspectFlags)vk::ImageAspectFlagBits::eColor,
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
			})
		);
	}

	void transitionImageLayout(
		VkImage image,
		vk::ImageLayout oldLayout,
		vk::ImageLayout newLayout,
		uint32_t mipLevels
	) const {
		VulkanSingleTimeCommand commandBuffer(
			device,
			graphicsQueue,
			(*this)()
		);

		auto barrier = vk::ImageMemoryBarrier(
			{},
			{},
			oldLayout,
			newLayout,
			VK_QUEUE_FAMILY_IGNORED,
			VK_QUEUE_FAMILY_IGNORED,
			image,
			vk::ImageSubresourceRange(
				vk::ImageAspectFlagBits::eColor,
				0,
				mipLevels,
				0,
				1
			)
		);

		vk::PipelineStageFlags sourceStage;
		vk::PipelineStageFlags destinationStage;

		if (oldLayout == vk::ImageLayout::eUndefined && 
			newLayout == vk::ImageLayout::eTransferDstOptimal
		) {
			barrier.srcAccessMask = {};
			barrier.dstAccessMask = vk::AccessFlagBits::eTransferWrite;

			sourceStage = vk::PipelineStageFlagBits::eTopOfPipe;
			destinationStage = vk::PipelineStageFlagBits::eTransfer;
		} else if (oldLayout == vk::ImageLayout::eTransferDstOptimal && 
			newLayout == vk::ImageLayout::eShaderReadOnlyOptimal
		) {
			barrier.srcAccessMask = vk::AccessFlagBits::eTransferWrite;
			barrier.dstAccessMask = vk::AccessFlagBits::eShaderRead;

			sourceStage = vk::PipelineStageFlagBits::eTransfer;
			destinationStage = vk::PipelineStageFlagBits::eFragmentShader;
		} else {
			throw Common::Exception() << "unsupported layout transition!";
		}

		commandBuffer().pipelineBarrier(
			sourceStage,
			destinationStage,
			{},
			{},
			{},
			barrier
		);
	}
};

namespace VulkanDeviceMakeFromStagingBuffer {
	std::pair<vk::Buffer, vk::DeviceMemory>
	create(
		vk::PhysicalDevice const physicalDevice,
		vk::Device const device,
		void const * const srcData,
		size_t bufferSize
	) {
		auto buffer = device.createBuffer(
			vk::BufferCreateInfo(VkBufferCreateInfo{
				.size = bufferSize,
				.usage = (VkBufferUsageFlags)vk::BufferUsageFlagBits::eTransferSrc,
				.sharingMode = (VkSharingMode)vk::SharingMode::eExclusive,
			})
		);
		auto memRequirements = device.getBufferMemoryRequirements(buffer);
		auto mem = device.allocateMemory(vk::MemoryAllocateInfo(VkMemoryAllocateInfo{
			.sType = (VkStructureType)vk::StructureType::eMemoryAllocateInfo,
			.allocationSize = memRequirements.size,
			.memoryTypeIndex = ThisVulkanPhysicalDevice::findMemoryType(
				physicalDevice,
				memRequirements.memoryTypeBits,
				vk::MemoryPropertyFlagBits::eHostVisible
				| vk::MemoryPropertyFlagBits::eHostCoherent
			),
		}));
		device.bindBufferMemory(
			buffer,
			mem,
			0);

		void * dstData = {};
		vkMapMemory(
			device,
			mem,
			0,
			bufferSize,
			0,
			&dstData
		);
		memcpy(dstData, srcData, (size_t)bufferSize);
		vkUnmapMemory(device, mem);
	
		return std::make_pair(buffer, mem);
	}
};

namespace VulkanDeviceMemoryBuffer  {
	std::pair<vk::Buffer, vk::DeviceMemory>
	create(
		vk::PhysicalDevice const physicalDevice,
		vk::Device const device,
		vk::DeviceSize size,
		vk::BufferUsageFlags usage,
		vk::MemoryPropertyFlags properties
	) {
		auto buffer = device.createBuffer(
			vk::BufferCreateInfo(VkBufferCreateInfo{
				.flags = (VkBufferCreateFlags)vk::BufferCreateFlags(),
				.size = size,
				.usage = (VkBufferUsageFlags)usage,
				.sharingMode = (VkSharingMode)vk::SharingMode::eExclusive,
			})
		);
		
		auto memRequirements = device.getBufferMemoryRequirements(buffer);
		auto memory = device.allocateMemory(vk::MemoryAllocateInfo(VkMemoryAllocateInfo{
			.sType = (VkStructureType)vk::StructureType::eMemoryAllocateInfo,
			.allocationSize = memRequirements.size,
			.memoryTypeIndex = ThisVulkanPhysicalDevice::findMemoryType(
				physicalDevice,
				memRequirements.memoryTypeBits,
				properties
			),
		}));
		device.bindBufferMemory(
			buffer,
			memory,
			0);
		return std::make_pair(buffer, memory);
	}

	std::pair<vk::Buffer, vk::DeviceMemory>
	makeBufferFromStaged(
		vk::PhysicalDevice const physicalDevice,
		vk::Device const & device,
		VulkanCommandPool const & commandPool,
		void const * const srcData,
		size_t bufferSize
	) {
		vk::Buffer stagingBuffer;
		vk::DeviceMemory stagingBufferMemory;
		std::tie(stagingBuffer, stagingBufferMemory) 
		= VulkanDeviceMakeFromStagingBuffer::create(
			physicalDevice,
			device,
			srcData,
			bufferSize
		);

		vk::Buffer buffer;
		vk::DeviceMemory memory;
		std::tie(buffer, memory) = VulkanDeviceMemoryBuffer::create(
			physicalDevice,
			device,
			bufferSize,
			vk::BufferUsageFlagBits::eTransferDst
			| vk::BufferUsageFlagBits::eVertexBuffer,
			vk::MemoryPropertyFlagBits::eDeviceLocal
		);
		
		commandPool.copyBuffer(
			stagingBuffer,
			buffer,
			bufferSize
		);

		device.destroyBuffer(stagingBuffer);
		device.freeMemory(stagingBufferMemory);

		return std::make_pair(buffer, memory);
	}
};

namespace VulkanDeviceMemoryImage {

	std::pair<vk::Image, vk::DeviceMemory>
	createImage(
		vk::PhysicalDevice const physicalDevice,
		vk::Device const & device,
		uint32_t width,
		uint32_t height,
		uint32_t mipLevels,
		vk::SampleCountFlagBits numSamples,
		vk::Format format,
		vk::ImageTiling tiling,
		vk::ImageUsageFlags usage,
		vk::MemoryPropertyFlags properties
	) {
		// TODO this as a ctor that just calls Super
		auto image = device.createImage(vk::ImageCreateInfo(VkImageCreateInfo{
			.sType = (VkStructureType)vk::StructureType::eImageCreateInfo,
			.imageType = (VkImageType)vk::ImageType::e2D,
			.format = (VkFormat)format,
			.extent = {
				.width = width,
				.height = height,
				.depth = 1,
			},
			.mipLevels = mipLevels,
			.arrayLayers = 1,
			.samples = (VkSampleCountFlagBits)numSamples,
			.tiling = (VkImageTiling)tiling,
			.usage = (VkImageUsageFlags)usage,
			.sharingMode = (VkSharingMode)vk::SharingMode::eExclusive,
			.initialLayout = (VkImageLayout)vk::ImageLayout::eUndefined,
		}));

		auto memRequirements = device.getImageMemoryRequirements(image);
		auto imageMemory = device.allocateMemory(vk::MemoryAllocateInfo(VkMemoryAllocateInfo{
			.sType = (VkStructureType)vk::StructureType::eMemoryAllocateInfo,
			.allocationSize = memRequirements.size,
			.memoryTypeIndex = ThisVulkanPhysicalDevice::findMemoryType(
				physicalDevice,
				memRequirements.memoryTypeBits,
				properties
			),
		}));
		device.bindImageMemory(image, imageMemory, 0);
		return std::make_pair(image, imageMemory);
	}

	std::pair<vk::Image, vk::DeviceMemory>
	makeTextureFromStaged(
		vk::PhysicalDevice physicalDevice,
		vk::Device device,
		VulkanCommandPool const & commandPool,
		void const * const srcData,
		size_t bufferSize,
		int texWidth,
		int texHeight,
		uint32_t mipLevels
	) {
		vk::Buffer stagingBuffer;
		vk::DeviceMemory stagingBufferMemory;
		std::tie(stagingBuffer, stagingBufferMemory) 
		= VulkanDeviceMakeFromStagingBuffer::create(
			physicalDevice,
			device,
			srcData,
			bufferSize
		);
		
		vk::Image image;
		vk::DeviceMemory imageMemory;
		std::tie(image, imageMemory) 
		= createImage(
			physicalDevice,
			device,
			texWidth,
			texHeight,
			mipLevels,
			vk::SampleCountFlagBits::e1,
			vk::Format::eR8G8B8A8Srgb,
			vk::ImageTiling::eOptimal,
			vk::ImageUsageFlagBits::eTransferSrc
			| vk::ImageUsageFlagBits::eTransferDst
			| vk::ImageUsageFlagBits::eSampled,
			vk::MemoryPropertyFlagBits::eDeviceLocal
		);

		commandPool.transitionImageLayout(
			image,
			vk::ImageLayout::eUndefined,
			vk::ImageLayout::eTransferDstOptimal,
			mipLevels
		);
		commandPool.copyBufferToImage(
			stagingBuffer,
			image,
			(uint32_t)texWidth,
			(uint32_t)texHeight
		);
		/*
		commandPool.transitionImageLayout(
			image,
			vk::ImageLayout::eTransferDstOptimal,
			vk::ImageLayout::eShaderReadOnlyOptimal,
			mipLevels
		);
		*/
		
		device.destroyBuffer(stagingBuffer);
		device.freeMemory(stagingBufferMemory);
		
		return std::make_pair(image, imageMemory);
	}

};

struct VulkanSwapChain {
protected:
	vk::SwapchainKHR obj;
	//owned
	vk::RenderPass renderPass;
	// hold for this class lifespan
	vk::Device device;

	vk::Image depthImage;
	vk::DeviceMemory depthImageMemory;
	vk::ImageView depthImageView;
	
	vk::Image colorImage;
	vk::DeviceMemory colorImageMemory;
	vk::ImageView colorImageView;
public:
	vk::Extent2D extent;
	
	// I would combine these into one struct so they can be dtored together
	// but it seems vulkan wants vk::Images linear for its getter?
	std::vector<vk::Image> images;
	std::vector<vk::ImageView> imageViews;
	std::vector<vk::Framebuffer> framebuffers;
	
public:
	auto operator()() const { return (VkSwapchainKHR)obj; }
	auto const & getRenderPass() const { return renderPass; }

	~VulkanSwapChain() {
		device.destroyImageView(depthImageView);
		device.freeMemory(depthImageMemory);
		device.destroyImage(depthImage);
		
		device.destroyImageView(colorImageView);
		device.freeMemory(colorImageMemory);
		device.destroyImage(colorImage);
		
		for (auto & framebuffer : framebuffers) {
			device.destroyFramebuffer(framebuffer);
		}
		for (auto & imageView : imageViews) {
			device.destroyImageView(imageView);
		}
		imageViews.clear();
		device.destroySwapchainKHR(obj);
	}

	// should this be a 'devices' or a 'swapchain' method?
	auto getImages() const {
		return device.getSwapchainImagesKHR(obj);
	}
	
	// ************** from here on down, app-specific **************
	// but so are all the member variables so ...

	VulkanSwapChain(
		Tensor::int2 screenSize,
		vk::PhysicalDevice physicalDevice,
		vk::Device device_,
		vk::SurfaceKHR surface,
		vk::SampleCountFlagBits msaaSamples
	) : device(device_) {
		auto swapChainSupport = ThisVulkanPhysicalDevice::querySwapChainSupport(physicalDevice, surface);
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
			.sType = (VkStructureType)vk::StructureType::eSwapchainCreateInfoKHR,
			.surface = surface,
			.minImageCount = imageCount,
			.imageFormat = (VkFormat)surfaceFormat.format,
			.imageColorSpace = (VkColorSpaceKHR)surfaceFormat.colorSpace,
			.imageExtent = extent,
			.imageArrayLayers = 1,
			.imageUsage = (VkImageUsageFlags)vk::ImageUsageFlagBits::eColorAttachment,
			.preTransform = (VkSurfaceTransformFlagBitsKHR)swapChainSupport.capabilities.currentTransform,
			.compositeAlpha = (VkCompositeAlphaFlagBitsKHR)vk::CompositeAlphaFlagBitsKHR::eOpaque,
			.presentMode = (VkPresentModeKHR)presentMode,
			.clipped = VK_TRUE,
		};
		auto indices = ThisVulkanPhysicalDevice::findQueueFamilies(physicalDevice, surface);
		auto queueFamilyIndices = Common::make_array<uint32_t>(
			(uint32_t)indices.graphicsFamily.value(),
			(uint32_t)indices.presentFamily.value()
		);
		if (indices.graphicsFamily != indices.presentFamily) {
			createInfo.imageSharingMode = (VkSharingMode)vk::SharingMode::eConcurrent;
			createInfo.queueFamilyIndexCount = (uint32_t)queueFamilyIndices.size();
			createInfo.pQueueFamilyIndices = queueFamilyIndices.data();
		} else {
			createInfo.imageSharingMode = (VkSharingMode)vk::SharingMode::eExclusive;
		}
		obj = device.createSwapchainKHR(createInfo);

		images = getImages();
		for (size_t i = 0; i < images.size(); i++) {
			imageViews.push_back(createImageView(
				images[i],
				surfaceFormat.format,
				vk::ImageAspectFlagBits::eColor,
				1
			));
		}
	
		renderPass = ThisVulkanRenderPass::create(
			physicalDevice,
			device,
			surfaceFormat.format,
			msaaSamples
		);
		
		//createColorResources
		auto colorFormat = surfaceFormat.format;
		
		std::tie(colorImage, colorImageMemory) 
		= VulkanDeviceMemoryImage::createImage(
			physicalDevice,
			device,
			extent.width,
			extent.height,
			1,
			msaaSamples,
			colorFormat,
			vk::ImageTiling::eOptimal,
			vk::ImageUsageFlagBits::eTransientAttachment | vk::ImageUsageFlagBits::eColorAttachment,
			vk::MemoryPropertyFlagBits::eDeviceLocal
		);
		colorImageView = createImageView(
			colorImage,
			colorFormat,
			vk::ImageAspectFlagBits::eColor,
			1
		);
		
		//createDepthResources
		auto depthFormat = ThisVulkanPhysicalDevice::findDepthFormat(physicalDevice);
		std::tie(depthImage, depthImageMemory) 
		= VulkanDeviceMemoryImage::createImage(
			physicalDevice,
			device,
			extent.width,
			extent.height,
			1,
			msaaSamples,
			depthFormat,
			vk::ImageTiling::eOptimal,
			vk::ImageUsageFlagBits::eDepthStencilAttachment,
			vk::MemoryPropertyFlagBits::eDeviceLocal
		);
		depthImageView = createImageView(
			depthImage,
			depthFormat,
			vk::ImageAspectFlagBits::eDepth,
			1
		);
		
		//createFramebuffers
		framebuffers.resize(imageViews.size());
		for (size_t i = 0; i < imageViews.size(); i++) {
			auto attachments = Common::make_array(
				(VkImageView)colorImageView,
				(VkImageView)depthImageView,
				(VkImageView)imageViews[i]
			);
			framebuffers[i] = device.createFramebuffer(vk::FramebufferCreateInfo(VkFramebufferCreateInfo{
				.sType = (VkStructureType)vk::StructureType::eFramebufferCreateInfo,
				.renderPass = renderPass,
				.attachmentCount = (uint32_t)attachments.size(),
				.pAttachments = attachments.data(),
				.width = extent.width,
				.height = extent.height,
				.layers = 1,
			}));
		}
	}

public:
	vk::ImageView createImageView(
		vk::Image image,
		vk::Format format,
		vk::ImageAspectFlags aspectFlags,
		uint32_t mipLevels
	) {
		return device.createImageView(vk::ImageViewCreateInfo(VkImageViewCreateInfo{
			.sType = (VkStructureType)vk::StructureType::eImageViewCreateInfo,
			.image = image,
			.viewType = (VkImageViewType)vk::ImageViewType::e2D,
			.format = (VkFormat)format,
			.subresourceRange = {
				.aspectMask = (VkImageAspectFlags)aspectFlags,
				.baseMipLevel = 0,
				.levelCount = mipLevels,
				.baseArrayLayer = 0,
				.layerCount = 1,
			},
		}));
	}

protected:
	static vk::SurfaceFormatKHR chooseSwapSurfaceFormat(
		std::vector<vk::SurfaceFormatKHR> const & availableFormats
	) {
		for (auto const & availableFormat : availableFormats) {
			if (availableFormat.format == vk::Format::eB8G8R8A8Srgb &&
				availableFormat.colorSpace == vk::ColorSpaceKHR::eSrgbNonlinear
			) {
				return availableFormat;
			}
		}
		return availableFormats[0];
	}

	static vk::PresentModeKHR chooseSwapPresentMode(
		std::vector<vk::PresentModeKHR> const & availablePresentModes
	) {
		for (auto const & availablePresentMode : availablePresentModes) {
			if (availablePresentMode == vk::PresentModeKHR::eMailbox) {
				return availablePresentMode;
			}
		}
		return vk::PresentModeKHR::eFifo;
	}

	static vk::Extent2D chooseSwapExtent(
		Tensor::int2 screenSize,
		vk::SurfaceCapabilitiesKHR const capabilities
	) {
		if (capabilities.currentExtent.width != std::numeric_limits<uint32_t>::max()) {
			return capabilities.currentExtent;
		} else {
			vk::Extent2D actualExtent(
				(uint32_t)screenSize.x,
				(uint32_t)screenSize.y
			);
			actualExtent.width = std::clamp(actualExtent.width, capabilities.minImageExtent.width, capabilities.maxImageExtent.width);
			actualExtent.height = std::clamp(actualExtent.height, capabilities.minImageExtent.height, capabilities.maxImageExtent.height);
			return actualExtent;
		}
	}
};

//only used by VulkanGraphicsPipeline
struct VulkanDescriptorSetLayout {
protected:
	vk::DescriptorSetLayout obj;
	//held for dtor
	vk::Device const device;
public:
	auto operator()() const { return (VkDescriptorSetLayout)obj; }

	~VulkanDescriptorSetLayout() {
		device.destroyDescriptorSetLayout(obj);
	}
	
	VulkanDescriptorSetLayout(
		vk::Device const device_
	) : device(device_) {
		auto bindings = Common::make_array(
			VkDescriptorSetLayoutBinding{	//uboLayoutBinding
				.binding = 0,
				.descriptorType = (VkDescriptorType)vk::DescriptorType::eUniformBuffer,
				.descriptorCount = 1,
				.stageFlags = (VkShaderStageFlags)vk::ShaderStageFlagBits::eVertex,
			},
			VkDescriptorSetLayoutBinding{	//samplerLayoutBinding
				.binding = 1,
				.descriptorType = (VkDescriptorType)vk::DescriptorType::eCombinedImageSampler,
				.descriptorCount = 1,
				.stageFlags = (VkShaderStageFlags)vk::ShaderStageFlagBits::eFragment,
			}
		);
		obj = device.createDescriptorSetLayout(vk::DescriptorSetLayoutCreateInfo(VkDescriptorSetLayoutCreateInfo{
			.sType = (VkStructureType)vk::StructureType::eDescriptorSetLayoutCreateInfo,
			.bindingCount = (uint32_t)bindings.size(),
			.pBindings = bindings.data(),
		}));
	}
};

// only wrapping crete/destroy so ... raii?
struct VulkanDescriptorPool {
protected:
	vk::DescriptorPool obj;
	//held for dtor
	vk::Device const device;
public:
	auto operator()() const { return (VkDescriptorPool)obj; }

	~VulkanDescriptorPool() {
		device.destroyDescriptorPool(obj);
	}

	VulkanDescriptorPool(
		vk::Device const device_,
		uint32_t const maxFramesInFlight
	) : device(device_) {
		auto poolSizes = Common::make_array(
			VkDescriptorPoolSize{
				.type = (VkDescriptorType)vk::DescriptorType::eUniformBuffer,
				.descriptorCount = maxFramesInFlight,
			},
			VkDescriptorPoolSize{
				.type = (VkDescriptorType)vk::DescriptorType::eCombinedImageSampler,
				.descriptorCount = maxFramesInFlight,
			}
		);
		obj = device.createDescriptorPool(vk::DescriptorPoolCreateInfo(VkDescriptorPoolCreateInfo{
			.sType = (VkStructureType)vk::StructureType::eDescriptorPoolCreateInfo,
			.maxSets = maxFramesInFlight,
			.poolSizeCount = (uint32_t)poolSizes.size(),
			.pPoolSizes = poolSizes.data(),
		}));
	}
};

//only used by VulkanGraphicsPipeline's ctor
struct VulkanShaderModule {
protected:
	vk::ShaderModule obj;
	//held:
	vk::Device const device;
public:
	auto operator()() const { return (VkShaderModule)obj; }

	~VulkanShaderModule() {
		device.destroyShaderModule(obj);
	}
	
	VulkanShaderModule(
		vk::Device const device_,
		std::string const code
	) : device(device_) {
		obj = device.createShaderModule(vk::ShaderModuleCreateInfo(VkShaderModuleCreateInfo{
			.sType = (VkStructureType)vk::StructureType::eShaderModuleCreateInfo,
			.codeSize = code.length(),
			.pCode = reinterpret_cast<uint32_t const *>(code.data()),
		}));
	}
};

struct VulkanGraphicsPipeline  {
protected:
	vk::Pipeline obj;
	//owned:
	vk::PipelineLayout pipelineLayout;
	std::unique_ptr<VulkanDescriptorSetLayout> descriptorSetLayout;
	
	//held:
	vk::Device const device;				//held for dtor
public:
	auto operator()() const { return (VkPipeline)obj; }
	auto const & getPipelineLayout() const { return pipelineLayout; }
	
	VulkanDescriptorSetLayout * getDescriptorSetLayout() { return descriptorSetLayout.get(); }
	VulkanDescriptorSetLayout const * getDescriptorSetLayout() const { return descriptorSetLayout.get(); }

	~VulkanGraphicsPipeline() {
		device.destroyPipelineLayout(pipelineLayout);
		device.destroyPipeline(obj);
		descriptorSetLayout = nullptr;
	}

	VulkanGraphicsPipeline(
		vk::PhysicalDevice const physicalDevice,
		vk::Device const device_,
		vk::RenderPass const renderPass,
		vk::SampleCountFlagBits msaaSamples
	) : device(device_) {
		
		// descriptorSetLayout is only used by graphicsPipeline
		descriptorSetLayout = std::make_unique<VulkanDescriptorSetLayout>(device);

		auto vertShaderModule = VulkanShaderModule(
			device,
			Common::File::read("shader-vert.spv")
		);
		
		auto fragShaderModule = VulkanShaderModule(
			device,
			Common::File::read("shader-frag.spv")
		);
		
		auto bindingDescriptions = Common::make_array(
			(VkVertexInputBindingDescription)Vertex::getBindingDescription()
		);
		auto attributeDescriptions = Vertex::getAttributeDescriptions();
		auto vertexInputInfo = VkPipelineVertexInputStateCreateInfo{
			.sType = (VkStructureType)vk::StructureType::ePipelineVertexInputStateCreateInfo,
			.vertexBindingDescriptionCount = (uint32_t)bindingDescriptions.size(),
			.pVertexBindingDescriptions = bindingDescriptions.data(),
			.vertexAttributeDescriptionCount = (uint32_t)attributeDescriptions.size(),
			.pVertexAttributeDescriptions = attributeDescriptions.data(),
		};

		auto inputAssembly = VkPipelineInputAssemblyStateCreateInfo{
			.sType = (VkStructureType)vk::StructureType::ePipelineInputAssemblyStateCreateInfo,
			.topology = (VkPrimitiveTopology)vk::PrimitiveTopology::eTriangleList,
			.primitiveRestartEnable = VK_FALSE,
		};

		auto viewportState = VkPipelineViewportStateCreateInfo{
			.sType = (VkStructureType)vk::StructureType::ePipelineViewportStateCreateInfo,
			.viewportCount = 1,
			.scissorCount = 1,
		};

		auto rasterizer = VkPipelineRasterizationStateCreateInfo{
			.sType = (VkStructureType)vk::StructureType::ePipelineRasterizationStateCreateInfo,
			.depthClampEnable = VK_FALSE,
			.rasterizerDiscardEnable = VK_FALSE,
			.polygonMode = (VkPolygonMode)vk::PolygonMode::eFill,
			//.cullMode = vk::CullModeFlagBits::eBack,
			//.frontFace = vk::FrontFace::eClockwise,
			//.frontFace = vk::FrontFace::eCounterClockwise,
			.depthBiasEnable = VK_FALSE,
			.lineWidth = 1,
		};

		auto multisampling = VkPipelineMultisampleStateCreateInfo{
			.sType = (VkStructureType)vk::StructureType::ePipelineMultisampleStateCreateInfo,
			.rasterizationSamples = (VkSampleCountFlagBits)msaaSamples,
			.sampleShadingEnable = VK_FALSE,
		};

		auto depthStencil = VkPipelineDepthStencilStateCreateInfo{
			.sType = (VkStructureType)vk::StructureType::ePipelineDepthStencilStateCreateInfo,
			.depthTestEnable = VK_TRUE,
			.depthWriteEnable = VK_TRUE,
			.depthCompareOp = (VkCompareOp)vk::CompareOp::eLess,
			.depthBoundsTestEnable = VK_FALSE,
			.stencilTestEnable = VK_FALSE,
		};

		auto colorBlendAttachment = VkPipelineColorBlendAttachmentState{
			.blendEnable = VK_FALSE,
			.colorWriteMask = (VkColorComponentFlags)(
				vk::ColorComponentFlagBits::eR
				| vk::ColorComponentFlagBits::eG
				| vk::ColorComponentFlagBits::eB
				| vk::ColorComponentFlagBits::eA
			),
		};

		auto colorBlending = VkPipelineColorBlendStateCreateInfo{
			.sType = (VkStructureType)vk::StructureType::ePipelineColorBlendStateCreateInfo,
			.logicOpEnable = VK_FALSE,
			.logicOp = (VkLogicOp)vk::LogicOp::eCopy,
			.attachmentCount = 1,
			.pAttachments = &colorBlendAttachment,
			.blendConstants = {0,0,0,0},
		};

		auto dynamicStates = Common::make_array<VkDynamicState>(
			(VkDynamicState)vk::DynamicState::eViewport,
			(VkDynamicState)vk::DynamicState::eScissor
		);
		auto dynamicState = VkPipelineDynamicStateCreateInfo{
			.sType = (VkStructureType)vk::StructureType::ePipelineDynamicStateCreateInfo,
			.dynamicStateCount = (uint32_t)dynamicStates.size(),
			.pDynamicStates = dynamicStates.data(),
		};
		
		auto descriptorSetLayouts = Common::make_array<VkDescriptorSetLayout>(
			(*descriptorSetLayout)()
		);
		pipelineLayout = device.createPipelineLayout(vk::PipelineLayoutCreateInfo(VkPipelineLayoutCreateInfo{
			.sType = (VkStructureType)vk::StructureType::ePipelineLayoutCreateInfo,
			.setLayoutCount = (uint32_t)descriptorSetLayouts.size(),
			.pSetLayouts = descriptorSetLayouts.data(),
		}));

		auto vertShaderStageInfo = VkPipelineShaderStageCreateInfo{
			.sType = (VkStructureType)vk::StructureType::ePipelineShaderStageCreateInfo,
			.stage = (VkShaderStageFlagBits)vk::ShaderStageFlagBits::eVertex,
			.module = vertShaderModule(),
			.pName = "main",
			//.pName = "vert",		// GLSL uses 'main', but clspv doesn't allow 'main', so ....
		};
		auto fragShaderStageInfo = VkPipelineShaderStageCreateInfo{
			.sType = (VkStructureType)vk::StructureType::ePipelineShaderStageCreateInfo,
			.stage = (VkShaderStageFlagBits)vk::ShaderStageFlagBits::eFragment,
			.module = fragShaderModule(),
			.pName = "main",
			//.pName = "frag",
		};
		auto shaderStages = Common::make_array(
			vertShaderStageInfo,
			fragShaderStageInfo
		);
		auto infos = Common::make_array(
			vk::GraphicsPipelineCreateInfo(VkGraphicsPipelineCreateInfo{
				.sType = (VkStructureType)vk::StructureType::eGraphicsPipelineCreateInfo,
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
				.renderPass = renderPass,
				.subpass = 0,
				.basePipelineHandle = VK_NULL_HANDLE,
			})
		);
		obj = device.createGraphicsPipelines(
			vk::PipelineCache{},
			infos 
		).value[0];
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

	vk::Instance instance;
	vk::DebugUtilsMessengerEXT debug;	// optional
	vk::SurfaceKHR surface;
	vk::Device device;
	vk::Queue graphicsQueue;
	vk::Queue presentQueue;
	std::unique_ptr<VulkanSwapChain> swapChain;
	std::unique_ptr<VulkanGraphicsPipeline> graphicsPipeline;
	std::unique_ptr<VulkanCommandPool> commandPool;
	
	vk::Buffer vertexBuffer;
	vk::DeviceMemory vertexBufferMemory;
	vk::Buffer indexBuffer;
	vk::DeviceMemory indexBufferMemory;
	
	uint32_t mipLevels = {};

	vk::Image textureImage;
	vk::DeviceMemory textureImageMemory;
	vk::ImageView textureImageView;
	vk::Sampler textureSampler;
	
	std::vector<Vertex> vertices;
	std::vector<uint32_t> indices;

	// hmm combine these two into a class?
	std::vector<std::pair<
		vk::Buffer,
		vk::DeviceMemory
	>> uniformBuffers;
	std::vector<void*> uniformBuffersMapped;
	
	std::unique_ptr<VulkanDescriptorPool> descriptorPool;
	
	// each of these, there are one per number of frames in flight
	std::vector<vk::DescriptorSet> descriptorSets;
	std::vector<vk::CommandBuffer> commandBuffers;
	std::vector<vk::Semaphore> imageAvailableSemaphores;
	std::vector<vk::Semaphore> renderFinishedSemaphores;
	std::vector<vk::Fence> inFlightFences;
	
	uint32_t currentFrame = {};
	
	bool framebufferResized = {};
public:
	void setFramebufferResized() { framebufferResized = true; }
protected:

	std::vector<char const *> const deviceExtensions = {
		VK_KHR_SWAPCHAIN_EXTENSION_NAME
	};

	//ok now we're at the point where we are recreating objects dependent on physicalDevice so
	vk::PhysicalDevice physicalDevice;
	vk::SampleCountFlagBits msaaSamples = vk::SampleCountFlagBits::e1;

public:
	VulkanCommon(::SDLApp::SDLApp const * const app_)
	: app(app_) {
		if (enableValidationLayers && !checkValidationLayerSupport()) {
			throw Common::Exception() << "validation layers requested, but not available!";
		}

		// hmm, maybe instance should be a shared_ptr and then passed to debug, surface, and physicalDevice ?
		instance = ThisVulkanInstance::create(app, enableValidationLayers);
		
		if (enableValidationLayers) {
			debug = ThisVulkanDebugMessenger::create(instance);
		}
		
		{
			VkSurfaceKHR h = {};
			SDL_VULKAN_SAFE(
				SDL_Vulkan_CreateSurface,
				app->getWindow(),
				instance,
				&h
			);
			surface = vk::SurfaceKHR(h);	
		}

		physicalDevice = ThisVulkanPhysicalDevice::create(
			instance,
			surface,
			deviceExtensions
		);
		msaaSamples = ThisVulkanPhysicalDevice::getMaxUsableSampleCount(physicalDevice);
		std::tie(device, graphicsQueue, presentQueue) = VulkanDevice::create(
			physicalDevice,
			surface,
			deviceExtensions,
			enableValidationLayers
		);
		swapChain = std::make_unique<VulkanSwapChain>(
			app->getScreenSize(),
			physicalDevice,
			device,
			surface,
			msaaSamples
		);
		graphicsPipeline = std::make_unique<VulkanGraphicsPipeline>(
			physicalDevice,
			device,
			swapChain->getRenderPass(),
			msaaSamples
		);
		
		{
			auto queueFamilyIndices = ThisVulkanPhysicalDevice::findQueueFamilies(physicalDevice, surface);
			commandPool = std::make_unique<VulkanCommandPool>(
				device,
				graphicsQueue,
				vk::CommandPoolCreateInfo(VkCommandPoolCreateInfo{
					.sType = (VkStructureType)vk::StructureType::eCommandPoolCreateInfo,
					.flags = (VkCommandPoolCreateFlags)vk::CommandPoolCreateFlagBits::eResetCommandBuffer,
					.queueFamilyIndex = queueFamilyIndices.graphicsFamily.value(),
				})
			);
		}
		
		createTextureImage();
	   
		textureImageView = swapChain->createImageView(
			textureImage,
			vk::Format::eR8G8B8A8Srgb,
			vk::ImageAspectFlagBits::eColor,
			mipLevels
		);

		textureSampler = device.createSampler(
			vk::SamplerCreateInfo(
				VkSamplerCreateInfo{
					.sType = (VkStructureType)vk::StructureType::eSamplerCreateInfo,
					.magFilter = (VkFilter)vk::Filter::eLinear,
					.minFilter = (VkFilter)vk::Filter::eLinear,
					.mipmapMode = (VkSamplerMipmapMode)vk::SamplerMipmapMode::eLinear,
					.addressModeU = (VkSamplerAddressMode)vk::SamplerAddressMode::eRepeat,
					.addressModeV = (VkSamplerAddressMode)vk::SamplerAddressMode::eRepeat,
					.addressModeW = (VkSamplerAddressMode)vk::SamplerAddressMode::eRepeat,
					.mipLodBias = 0,
					.anisotropyEnable = VK_TRUE,
					.maxAnisotropy = physicalDevice.getProperties().limits.maxSamplerAnisotropy,
					.compareEnable = VK_FALSE,
					.compareOp = (VkCompareOp)vk::CompareOp::eAlways,
					.minLod = 0,
					.maxLod = static_cast<float>(mipLevels),
					.borderColor = (VkBorderColor)vk::BorderColor::eIntOpaqueBlack,
					.unnormalizedCoordinates = VK_FALSE,
				}
			)
		);

		loadModel();
		
		std::tie(vertexBuffer, vertexBufferMemory) 
		= VulkanDeviceMemoryBuffer::makeBufferFromStaged(
			physicalDevice,
			device,
			*commandPool.get(),
			vertices.data(),
			sizeof(vertices[0]) * vertices.size()
		);

		std::tie(indexBuffer, indexBufferMemory)
		= VulkanDeviceMemoryBuffer::makeBufferFromStaged(
			physicalDevice,
			device,
			*commandPool.get(),
			indices.data(),
			sizeof(indices[0]) * indices.size()
		);

		uniformBuffersMapped.resize(maxFramesInFlight);
		for (size_t i = 0; i < maxFramesInFlight; i++) {
			uniformBuffers.push_back(VulkanDeviceMemoryBuffer::create(
				physicalDevice,
				device,
				sizeof(UniformBufferObject),
				vk::BufferUsageFlagBits::eUniformBuffer,
				vk::MemoryPropertyFlagBits::eHostVisible
				| vk::MemoryPropertyFlagBits::eHostCoherent
			));
			vkMapMemory(
				device,
				std::get<1>(uniformBuffers[i]),
				0,
				sizeof(UniformBufferObject),
				0,
				&uniformBuffersMapped[i]
			);
		}

		descriptorPool = std::make_unique<VulkanDescriptorPool>(
			device,
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
		device.freeMemory(vertexBufferMemory);
		device.destroyBuffer(vertexBuffer);
		device.freeMemory(indexBufferMemory);
		device.destroyBuffer(indexBuffer);
		for (auto & o : uniformBuffers) {
			device.freeMemory(std::get<1>(o));
			device.destroyBuffer(std::get<0>(o));
		}

		device.destroySampler(textureSampler);
		device.destroyImageView(textureImageView);
		device.freeMemory(textureImageMemory);
		device.destroyImage(textureImage);
		for (size_t i = 0; i < maxFramesInFlight; i++) {
			device.destroySemaphore(renderFinishedSemaphores[i]);
			device.destroySemaphore(imageAvailableSemaphores[i]);
			device.destroyFence(inFlightFences[i]);
		}
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
		vk::DeviceSize const bufferSize = texSize.x * texSize.y * texBPP;
		mipLevels = (uint32_t)std::floor(std::log2(std::max(texSize.x, texSize.y))) + 1;
	
		std::tie(textureImage, textureImageMemory) 
		= VulkanDeviceMemoryImage::makeTextureFromStaged(
			physicalDevice,
			device,
			*commandPool.get(),
			srcData,
			bufferSize,
			texSize.x,
			texSize.y,
			mipLevels
		);
	
		generateMipmaps(
			textureImage,
			vk::Format::eR8G8B8A8Srgb,
			texSize.x,
			texSize.y,
			mipLevels
		);
	}

	void generateMipmaps(
		vk::Image image,
		vk::Format imageFormat,
		int32_t texWidth,
		int32_t texHeight,
		uint32_t mipLevels
	) {
		// Check if image format supports linear blitting
		auto formatProperties = physicalDevice.getFormatProperties(imageFormat);

		if (!(formatProperties.optimalTilingFeatures & vk::FormatFeatureFlagBits::eSampledImageFilterLinear)) {
			throw Common::Exception() << "texture image format does not support linear blitting!";
		}

		VulkanSingleTimeCommand commandBuffer(
			device,
			graphicsQueue,
			(*commandPool)()
		);

		auto barrier = vk::ImageMemoryBarrier(VkImageMemoryBarrier{
			.sType = (VkStructureType)vk::StructureType::eImageMemoryBarrier,
			.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
			.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
			.image = (VkImage)image,
			.subresourceRange = {
				.aspectMask = (VkImageAspectFlags)vk::ImageAspectFlagBits::eColor,
				.levelCount = 1,
				.baseArrayLayer = 0,
				.layerCount = 1,
			},
		});
		
		int32_t mipWidth = texWidth;
		int32_t mipHeight = texHeight;

		for (uint32_t i = 1; i < mipLevels; i++) {
			barrier.subresourceRange.baseMipLevel = i - 1;
			barrier.oldLayout = vk::ImageLayout::eTransferDstOptimal;
			barrier.newLayout = vk::ImageLayout::eTransferSrcOptimal;
			barrier.srcAccessMask = vk::AccessFlagBits::eTransferWrite;
			barrier.dstAccessMask = vk::AccessFlagBits::eTransferRead;

			commandBuffer().pipelineBarrier(
				vk::PipelineStageFlagBits::eTransfer,
				vk::PipelineStageFlagBits::eTransfer,
				vk::DependencyFlags{},
				{},
				{},
				barrier
			);

			auto blit = vk::ImageBlit(VkImageBlit{
				.srcSubresource = {
					.aspectMask = (VkImageAspectFlags)vk::ImageAspectFlagBits::eColor,
					.mipLevel = i - 1,
					.baseArrayLayer = 0,
					.layerCount = 1,
				},
				.srcOffsets = {
					{0, 0, 0},
					{mipWidth, mipHeight, 1},
				},
				.dstSubresource = {
					.aspectMask = (VkImageAspectFlags)vk::ImageAspectFlagBits::eColor,
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
			});
			
			commandBuffer().blitImage(
				image,
				vk::ImageLayout::eTransferSrcOptimal,
				image,
				vk::ImageLayout::eTransferDstOptimal,
				Common::make_array(blit),
				vk::Filter::eLinear
			);

			barrier.oldLayout = vk::ImageLayout::eTransferSrcOptimal;
			barrier.newLayout = vk::ImageLayout::eShaderReadOnlyOptimal;
			barrier.srcAccessMask = vk::AccessFlagBits::eTransferRead;
			barrier.dstAccessMask = vk::AccessFlagBits::eShaderRead;

			commandBuffer().pipelineBarrier(
				vk::PipelineStageFlagBits::eTransfer,
				vk::PipelineStageFlagBits::eFragmentShader,
				{},
				{},
				{},
				barrier
			);

			if (mipWidth > 1) mipWidth /= 2;
			if (mipHeight > 1) mipHeight /= 2;
		}

		barrier.subresourceRange.baseMipLevel = mipLevels - 1;
		barrier.oldLayout = vk::ImageLayout::eTransferDstOptimal;
		barrier.newLayout = vk::ImageLayout::eShaderReadOnlyOptimal;
		barrier.srcAccessMask = vk::AccessFlagBits::eTransferWrite;
		barrier.dstAccessMask = vk::AccessFlagBits::eShaderRead;

		commandBuffer().pipelineBarrier(
			vk::PipelineStageFlagBits::eTransfer,
			vk::PipelineStageFlagBits::eFragmentShader,
			{},
			{},
			{},
			barrier
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
		if (!app->getScreenSize().x || !app->getScreenSize().y) {
			throw Common::Exception() << "here";
		}
#endif
		device.waitIdle();
		swapChain = std::make_unique<VulkanSwapChain>(
			app->getScreenSize(),
			physicalDevice,
			device,
			surface,
			msaaSamples
		);
	}

	void initCommandBuffers() {
		// TODO this matches 'VulkanSingleTimeCommand' ctor
		commandBuffers = device.allocateCommandBuffers(vk::CommandBufferAllocateInfo(VkCommandBufferAllocateInfo{
			.sType = (VkStructureType)vk::StructureType::eCommandBufferAllocateInfo,
			.commandPool = (*commandPool)(),
			.level = (VkCommandBufferLevel)vk::CommandBufferLevel::ePrimary,
			.commandBufferCount = (uint32_t)maxFramesInFlight,
		}));
		// end part that matches
	}

	void recordCommandBuffer(
		vk::CommandBuffer const commandBuffer,
		uint32_t imageIndex
	) {
		// TODO this part matches VulkanSingleTimeCommand ctor
		commandBuffer.begin(VkCommandBufferBeginInfo{
			.sType = (VkStructureType)vk::StructureType::eCommandBufferBeginInfo,
		});
		// end part that matches

		auto clearValues = Common::make_array(
			VkClearValue{
				.color = {{0, 0, 0, 1}},
			},
			VkClearValue{
				.depthStencil = {1, 0},
			}
		);
		commandBuffer.beginRenderPass(
			vk::RenderPassBeginInfo(VkRenderPassBeginInfo{
				.sType = (VkStructureType)vk::StructureType::eRenderPassBeginInfo,
				.renderPass = swapChain->getRenderPass(),
				.framebuffer = swapChain->framebuffers[imageIndex],
				.renderArea = {
					.offset = {0, 0},
					.extent = swapChain->extent,
				},
				.clearValueCount = (uint32_t)clearValues.size(),
				.pClearValues = clearValues.data(),
			}),
			vk::SubpassContents::eInline
		);

		{
			commandBuffer.bindPipeline(
				vk::PipelineBindPoint::eGraphics,
				(*graphicsPipeline)()
			);

			commandBuffer.setViewport(
				0,
				Common::make_array(
					vk::Viewport(VkViewport{
						.x = 0,
						.y = 0,
						.width = (float)swapChain->extent.width,
						.height = (float)swapChain->extent.height,
						.minDepth = 0,
						.maxDepth = 1,
					})
				)
			);

			commandBuffer.setScissor(
				0,
				Common::make_array(
					vk::Rect2D(VkRect2D{
						.offset = {0, 0},
						.extent = swapChain->extent,
					})
				)
			);

			commandBuffer.bindVertexBuffers(
				0,
				Common::make_array(
					vertexBuffer
				),
				Common::make_array<vk::DeviceSize>(0)
			);

			commandBuffer.bindIndexBuffer(
				indexBuffer,
				0,
				vk::IndexType::eUint32
			);

			commandBuffer.bindDescriptorSets(
				vk::PipelineBindPoint::eGraphics,
				graphicsPipeline->getPipelineLayout(),
				0,
				descriptorSets[currentFrame],
				{}
			);

			commandBuffer.drawIndexed(
				(uint32_t)indices.size(),
				1,
				0,
				0,
				0
			);
		}

		commandBuffer.endRenderPass();
		commandBuffer.end();
	}

	void initSyncObjects() {
		imageAvailableSemaphores.resize(maxFramesInFlight);
		renderFinishedSemaphores.resize(maxFramesInFlight);
		inFlightFences.resize(maxFramesInFlight);

		auto semaphoreInfo = VkSemaphoreCreateInfo{
			.sType = (VkStructureType)vk::StructureType::eSemaphoreCreateInfo,
		};

		auto fenceInfo = VkFenceCreateInfo{
			.sType = (VkStructureType)vk::StructureType::eFenceCreateInfo,
			.flags = (VkFenceCreateFlags)vk::FenceCreateFlagBits::eSignaled,
		};

		for (size_t i = 0; i < maxFramesInFlight; i++) {
			imageAvailableSemaphores[i] = device.createSemaphore(semaphoreInfo);
			renderFinishedSemaphores[i] = device.createSemaphore(semaphoreInfo);
			inFlightFences[i] = device.createFence(fenceInfo);
		}
	}

	void createDescriptorSets() {
		std::vector<VkDescriptorSetLayout> layouts(maxFramesInFlight, (*graphicsPipeline->getDescriptorSetLayout())());
		descriptorSets = device.allocateDescriptorSets(
			vk::DescriptorSetAllocateInfo(VkDescriptorSetAllocateInfo{
				.sType = (VkStructureType)vk::StructureType::eDescriptorSetAllocateInfo,
				.descriptorPool = (*descriptorPool)(),
				.descriptorSetCount = (uint32_t)maxFramesInFlight,
				.pSetLayouts = layouts.data(),
			})
		);

		for (size_t i = 0; i < maxFramesInFlight; i++) {
			auto bufferInfo = VkDescriptorBufferInfo{
				.buffer = std::get<0>(uniformBuffers[i]),
				.offset = 0,
				.range = sizeof(UniformBufferObject),
			};
			auto imageInfo = VkDescriptorImageInfo{
				.sampler = textureSampler,
				.imageView = textureImageView,
				.imageLayout = (VkImageLayout)vk::ImageLayout::eShaderReadOnlyOptimal,
			};
			auto descriptorWrites = Common::make_array(
				vk::WriteDescriptorSet(VkWriteDescriptorSet{
					.sType = (VkStructureType)vk::StructureType::eWriteDescriptorSet,
					.dstSet = descriptorSets[i],
					.dstBinding = 0,
					.descriptorCount = 1,
					.descriptorType = (VkDescriptorType)vk::DescriptorType::eUniformBuffer,
					.pBufferInfo = &bufferInfo,
				}),
				vk::WriteDescriptorSet(VkWriteDescriptorSet{
					.sType = (VkStructureType)vk::StructureType::eWriteDescriptorSet,
					.dstSet = descriptorSets[i],
					.dstBinding = 1,
					.descriptorCount = 1,
					.descriptorType = (VkDescriptorType)vk::DescriptorType::eCombinedImageSampler,
					.pImageInfo = &imageInfo,
				})
			);
			device.updateDescriptorSets(
				descriptorWrites,
				{}
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
		{
			auto fences = Common::make_array<vk::Fence>(
				(vk::Fence)inFlightFences[currentFrame]
			);
			auto result = device.waitForFences(
				fences,
				VK_TRUE,
				UINT64_MAX
			);
			if (result != vk::Result::eSuccess) throw Common::Exception() << "" << (int)result;
		}

		// should this be a swapChain method?
		// if so then how to return both imageIndex and result?
		// if not result then how to provide callback upon recreate-swap-chain?
		uint32_t imageIndex = {};
		{
			auto result = vk::Result(vkAcquireNextImageKHR(
				device,
				(*swapChain)(),
				UINT64_MAX,
				imageAvailableSemaphores[currentFrame],
				VK_NULL_HANDLE,
				&imageIndex
			));
			if (result == vk::Result::eErrorOutOfDateKHR) {
				recreateSwapChain();
				return;
			} else if (
				result != vk::Result::eSuccess && 
				result != vk::Result::eSuboptimalKHR
			) {
				throw Common::Exception() << "vkAcquireNextImageKHR failed: " << result;
			}
		}

		updateUniformBuffer(currentFrame);

		device.resetFences(
			Common::make_array(
				inFlightFences[currentFrame]
			)
		);

		{
			auto o = commandBuffers[currentFrame];
			o.reset();
			recordCommandBuffer(o, imageIndex);
		}

		auto waitSemaphores = Common::make_array(
			(VkSemaphore)imageAvailableSemaphores[currentFrame]
		);
		auto waitStages = Common::make_array<VkPipelineStageFlags>(
			(VkPipelineStageFlags)vk::PipelineStageFlagBits::eColorAttachmentOutput
		);
		static_assert(waitSemaphores.size() == waitStages.size());
		
		auto signalSemaphores = Common::make_array(
			(VkSemaphore)renderFinishedSemaphores[currentFrame]
		);

		// static assert sizes match?
		auto tmpCmdBuf = (VkCommandBuffer)commandBuffers[currentFrame];
		graphicsQueue.submit(
			vk::SubmitInfo(VkSubmitInfo{
				.sType = (VkStructureType)vk::StructureType::eSubmitInfo,
				.waitSemaphoreCount = (uint32_t)waitSemaphores.size(),
				.pWaitSemaphores = waitSemaphores.data(),
				.pWaitDstStageMask = waitStages.data(),
				.commandBufferCount = 1,
				.pCommandBuffers = &tmpCmdBuf,
				.signalSemaphoreCount = (uint32_t)signalSemaphores.size(),
				.pSignalSemaphores = signalSemaphores.data(),
			}),
			inFlightFences[currentFrame]
		);
		
		auto swapChains = Common::make_array<VkSwapchainKHR>(
			(*swapChain)()
		);
		auto result = presentQueue.presentKHR(
			vk::PresentInfoKHR(VkPresentInfoKHR{
				.sType = (VkStructureType)vk::StructureType::ePresentInfoKHR,
				.waitSemaphoreCount = (uint32_t)signalSemaphores.size(),
				// these two sizes need t match (right?)
				.pWaitSemaphores = signalSemaphores.data(),
				.swapchainCount = (uint32_t)swapChains.size(),
				// wait do these two sizes need to match?
				.pSwapchains = swapChains.data(),
				.pImageIndices = &imageIndex,
			})
		);
		if (result == vk::Result::eErrorOutOfDateKHR || 
			result == vk::Result::eSuboptimalKHR || 
			framebufferResized
		) {
			framebufferResized = false;
			recreateSwapChain();
		} else if (result != vk::Result::eSuccess) {
			throw Common::Exception() << "vkQueuePresentKHR failed: " << result;
		}

		currentFrame = (currentFrame + 1) % maxFramesInFlight;
	}
public:
	void loopDone() {
		device.waitIdle();
	}
};

struct Test : public ::SDLApp::SDLApp {
	using Super = ::SDLApp::SDLApp;

protected:
	std::unique_ptr<VulkanCommon> vkCommon;
	
	virtual void initWindow() {
		Super::initWindow();
		vkCommon = std::make_unique<VulkanCommon>(this);
	}

public:
	virtual std::string getTitle() const {
		return "Vulkan Test";
	}

protected:
	virtual Uint32 getSDLCreateWindowFlags() {
		auto flags = Super::getSDLCreateWindowFlags();
		flags |= SDL_WINDOW_VULKAN;
//		flags &= ~SDL_WINDOW_RESIZABLE;
		return flags;
	}

	virtual void loop() {
		Super::loop();
		//why here instead of shutdown?

		vkCommon->loopDone();
	}
	
	virtual void onUpdate() {
		Super::onUpdate();
		vkCommon->drawFrame();
	}

	virtual void onResize() {
		vkCommon->setFramebufferResized();
	}
};

SDLAPP_MAIN(Test)
