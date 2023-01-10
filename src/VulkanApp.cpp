#include "SDLApp/SDLApp.h"
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

namespace Common {

// wait .. shouldn't I just make this work with Function?
// shouldn't Function accept <FuncType> by default, and then overload?
template<typename FuncType>
struct FunctionPointer;

// TODO can I skip FunctionPointer altogether and just use Function<something_t<F>> ?
template<typename Return_, typename... ArgList>
struct FunctionPointer<Return_ (*)(ArgList...)> : public Function<Return_(ArgList...)> {};

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
	Tensor::float4x4 model;
	Tensor::float4x4 view;
	Tensor::float4x4 proj;
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


struct VulkanInstance {
protected:
	VkInstance handle = {};
public:
	decltype(handle) operator()() const { return handle; }

	~VulkanInstance() {
		if (handle) vkDestroyInstance(handle, nullptr);
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

struct VulkanDebugMessenger {
protected:
	VkDebugUtilsMessengerEXT handle = {};
	VulkanInstance const * const instance = {};	//from VulkanCommon, needs to be accessed for dtor to work
	
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
			vkDestroyDebugUtilsMessengerEXT((*instance)(), handle, nullptr);
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

struct VulkanSurface {
protected:
	VkSurfaceKHR handle;
	VkInstance instance;	//from VulkanCommon, needs to be held for dtor to work
public:
	decltype(handle) operator()() const { return handle; }
	
	~VulkanSurface() {
		if (handle) vkDestroySurfaceKHR(instance, handle, nullptr);
	}

	VulkanSurface(
		SDL_Window * const window,
		VkInstance instance_
	) : instance(instance_) {
		// https://vulkan-tutorial.com/Drawing_a_triangle/Presentation/Window_surface
		SDL_VULKAN_SAFE(SDL_Vulkan_CreateSurface, window, instance, &handle);
	}
};

struct VulkanPhysicalDevice {
protected:
	VkPhysicalDevice handle = {};
public:
	decltype(handle) operator()() const { return handle; }
	
	VulkanPhysicalDevice(VkPhysicalDevice handle_)
	: handle(handle_) 
	{}

	std::vector<VkQueueFamilyProperties> getQueueFamilyProperties() const {
		return vulkanEnum<VkQueueFamilyProperties>(
			NAME_PAIR(vkGetPhysicalDeviceQueueFamilyProperties),
			handle
		);
	}

	VkPhysicalDeviceProperties getProperties() const {
		VkPhysicalDeviceProperties physDevProps;
		vkGetPhysicalDeviceProperties(handle, &physDevProps);
		return physDevProps;
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

	// this is halfway app-specific.  it was in the tut's organization but I'm 50/50 if it is good design

public:
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

struct VulkanLogicalDevice {
protected:
	VkDevice handle = {};
public:
	VkQueue graphicsQueue = {};
	VkQueue presentQueue = {};

public:
	decltype(handle) operator()() const { return handle; }
	
	~VulkanLogicalDevice() {
		if (handle) vkDestroyDevice(handle, nullptr);
	}
	
	VkQueue getQueue(
		uint32_t queueFamilyIndex,
		uint32_t queueIndex = 0
	) const {
		VkQueue result;
		vkGetDeviceQueue(handle, queueFamilyIndex, queueIndex, &result);
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
		VULKAN_SAFE(vkCreateDevice, (*physicalDevice)(), &createInfo, nullptr, &handle);
	
		graphicsQueue = getQueue(indices.graphicsFamily.value());
		presentQueue = getQueue(indices.presentFamily.value());
	}
};

struct VulkanRenderPass {
protected:
	//owned
	VkRenderPass handle = {};
	//held
	VkDevice device = {};
public:
	decltype(handle) operator()() const { return handle; }
	
	~VulkanRenderPass() {
		if (handle) vkDestroyRenderPass(device, handle, nullptr);
	}
	
	// ************** from here on down, app-specific **************  

	VulkanRenderPass(
		VkDevice device_,
		VkFormat swapChainImageFormat
	) : device(device_) {
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
		VULKAN_SAFE(vkCreateRenderPass, device, &renderPassInfo, nullptr, &handle);
	}
};

struct VulkanSwapChain {
protected:
	VkSwapchainKHR handle = {};
	std::unique_ptr<VulkanRenderPass> renderPass;
public:
	VkExtent2D extent;
	
	// I would combine these into one struct so they can be dtored together
	// but it seems vulkan wants VkImages linear for its getter?
	std::vector<VkImage> images;
	std::vector<VkImageView> imageViews;
	std::vector<VkFramebuffer> framebuffers;
	
protected:
	// hold for this class lifespan
	VkDevice device;

public:
	decltype(handle) operator()() const { return handle; }
	VkRenderPass getRenderPass() const { return (*renderPass)(); }

	~VulkanSwapChain() {
		for (auto framebuffer : framebuffers) {
			vkDestroyFramebuffer(device, framebuffer, nullptr);
		}
		renderPass = nullptr;
		for (auto imageView : imageViews) {
			vkDestroyImageView(device, imageView, nullptr);
		}
		if (handle) vkDestroySwapchainKHR(device, handle, nullptr);
	}
	
	// ************** from here on down, app-specific **************  
	
	VulkanSwapChain(
		Tensor::int2 screenSize,
		VulkanPhysicalDevice * const physicalDevice,
		VkDevice device_,
		VkSurfaceKHR surface
	) : device(device_) {
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
		VULKAN_SAFE(vkCreateSwapchainKHR, device, &createInfo, nullptr, &handle);

		images = vulkanEnum<VkImage>(NAME_PAIR(vkGetSwapchainImagesKHR), device, handle);

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
			VULKAN_SAFE(vkCreateImageView, device, &createInfo, nullptr, &imageViews[i]);
		}
	
		renderPass = std::make_unique<VulkanRenderPass>(device, surfaceFormat.format);

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
			VULKAN_SAFE(vkCreateFramebuffer, device, &framebufferInfo, nullptr, &framebuffers[i]);
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

struct VulkanDescriptorSetLayout {
protected:
	VkDescriptorSetLayout handle = {};	//owned
	VkDevice device = {};				//held for dtor
public:
	decltype(handle) operator()() const { return handle; }

	~VulkanDescriptorSetLayout() {
		if (handle) {
			vkDestroyDescriptorSetLayout(device, handle, nullptr);
		}
	}
	
	VulkanDescriptorSetLayout(
		VkDevice device_
	) : device(device_) {
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
		VULKAN_SAFE(vkCreateDescriptorSetLayout, device, &layoutInfo, nullptr, &handle);
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
	std::unique_ptr<VulkanDescriptorSetLayout> descriptorSetLayout;
	
	VkPipelineLayout pipelineLayout = {};
	VkPipeline graphicsPipeline = {};
	VkCommandPool commandPool = {};
	
	VkBuffer vertexBuffer = {};
	VkDeviceMemory vertexBufferMemory = {};
	VkBuffer indexBuffer = {};
	VkDeviceMemory indexBufferMemory = {};

	std::vector<VkBuffer> uniformBuffers;
	std::vector<VkDeviceMemory> uniformBuffersMemory;
	std::vector<void*> uniformBuffersMapped;

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
			(*device)(),
			(*surface)()
		);
		
		descriptorSetLayout = std::make_unique<VulkanDescriptorSetLayout>((*device)());
		
		initGraphicsPipeline();
		initCommandPool(physicalDevice.get());
		initVertexBuffer();
		initIndexBuffer();
		initUniformBuffers();
		initCommandBuffers();
		initSyncObjects();
	}

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

	~VulkanCommon() {
		swapChain = nullptr;
		
		if (graphicsPipeline) vkDestroyPipeline((*device)(), graphicsPipeline, nullptr);
		if (pipelineLayout) vkDestroyPipelineLayout((*device)(), pipelineLayout, nullptr);

		for (size_t i = 0; i < MAX_FRAMES_IN_FLIGHT; i++) {
			vkDestroyBuffer((*device)(), uniformBuffers[i], nullptr);
			vkFreeMemory((*device)(), uniformBuffersMemory[i], nullptr);
		}

		descriptorSetLayout = nullptr;

		vkDestroyBuffer((*device)(), indexBuffer, nullptr);
		vkFreeMemory((*device)(), indexBufferMemory, nullptr);
		
		vkDestroyBuffer((*device)(), vertexBuffer, nullptr);
		vkFreeMemory((*device)(), vertexBufferMemory, nullptr);

		for (size_t i = 0; i < MAX_FRAMES_IN_FLIGHT; i++) {
			vkDestroySemaphore((*device)(), renderFinishedSemaphores[i], nullptr);
			vkDestroySemaphore((*device)(), imageAvailableSemaphores[i], nullptr);
			vkDestroyFence((*device)(), inFlightFences[i], nullptr);
		}

		vkDestroyCommandPool((*device)(), commandPool, nullptr);
		
		device = nullptr;
		surface = nullptr;
		debug = nullptr;
		instance = nullptr;
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
		vkDeviceWaitIdle((*device)());

		swapChain = std::make_unique<VulkanSwapChain>(
			app->getScreenSize(),
			physicalDevice.get(),
			(*device)(),
			(*surface)()
		);
	}
	
	void initGraphicsPipeline() {
		auto vertShaderCode = Common::File::read("shader-vert.spv");
		auto fragShaderCode = Common::File::read("shader-frag.spv");

		VkShaderModule vertShaderModule = createShaderModule(vertShaderCode);
		VkShaderModule fragShaderModule = createShaderModule(fragShaderCode);

		VkPipelineShaderStageCreateInfo vertShaderStageInfo = {};
		vertShaderStageInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
		vertShaderStageInfo.stage = VK_SHADER_STAGE_VERTEX_BIT;
		vertShaderStageInfo.module = vertShaderModule;
		// GLSL uses 'main', but clspv doesn't allow 'main', so ....
		vertShaderStageInfo.pName = "main";
		//vertShaderStageInfo.pName = "vert";

		VkPipelineShaderStageCreateInfo fragShaderStageInfo = {};
		fragShaderStageInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
		fragShaderStageInfo.stage = VK_SHADER_STAGE_FRAGMENT_BIT;
		fragShaderStageInfo.module = fragShaderModule;
		fragShaderStageInfo.pName = "main";
		//fragShaderStageInfo.pName = "frag";

		VkPipelineVertexInputStateCreateInfo vertexInputInfo = {};
		vertexInputInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO;

		auto bindingDescription = Vertex::getBindingDescription();
		vertexInputInfo.vertexBindingDescriptionCount = 1;
		vertexInputInfo.pVertexBindingDescriptions = &bindingDescription;
		
		auto attributeDescriptions = Vertex::getAttributeDescriptions();
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
		rasterizer.frontFace = VK_FRONT_FACE_CLOCKWISE;
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

		std::vector<VkDescriptorSetLayout> descriptorSetLayouts = {
			(*descriptorSetLayout)(),
		};

		VkPipelineLayoutCreateInfo pipelineLayoutInfo = {};
		pipelineLayoutInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
		pipelineLayoutInfo.setLayoutCount = static_cast<uint32_t>(descriptorSetLayouts.size());
		pipelineLayoutInfo.pSetLayouts = descriptorSetLayouts.data();
		VULKAN_SAFE(vkCreatePipelineLayout, (*device)(), &pipelineLayoutInfo, nullptr, &pipelineLayout);

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
		pipelineInfo.renderPass = swapChain->getRenderPass();
		pipelineInfo.subpass = 0;
		pipelineInfo.basePipelineHandle = VK_NULL_HANDLE;
		VULKAN_SAFE(vkCreateGraphicsPipelines, (*device)(), (VkPipelineCache)VK_NULL_HANDLE, 1, &pipelineInfo, nullptr, &graphicsPipeline);

		vkDestroyShaderModule((*device)(), fragShaderModule, nullptr);
		vkDestroyShaderModule((*device)(), vertShaderModule, nullptr);
	}

	VkShaderModule createShaderModule(std::string const & code) {
		VkShaderModuleCreateInfo createInfo = {};
		createInfo.sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO;
		createInfo.codeSize = code.length();
		createInfo.pCode = reinterpret_cast<const uint32_t*>(code.data());

		VkShaderModule shaderModule;
		VULKAN_SAFE(vkCreateShaderModule, (*device)(), &createInfo, nullptr, &shaderModule);
		return shaderModule;
	}

	void initCommandPool(VulkanPhysicalDevice const * physicalDevice) {
		auto queueFamilyIndices = physicalDevice->findQueueFamilies((*surface)());

		VkCommandPoolCreateInfo poolInfo = {};
		poolInfo.sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO;
		poolInfo.flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;
		poolInfo.queueFamilyIndex = queueFamilyIndices.graphicsFamily.value();

		VULKAN_SAFE(vkCreateCommandPool, (*device)(), &poolInfo, nullptr, &commandPool);
	}

	void initVertexBuffer() {
		VkDeviceSize bufferSize = sizeof(vertices[0]) * vertices.size();

		VkBuffer stagingBuffer;
		VkDeviceMemory stagingBufferMemory;
		std::tie(stagingBuffer, stagingBufferMemory) = initBuffer(
			bufferSize,
			VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
			VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT
		);

		void * data = {};
		vkMapMemory(
			(*device)(),
			stagingBufferMemory,
			0,
			bufferSize,
			0,
			&data
		);
		memcpy(data, vertices.data(), (size_t)bufferSize);
		vkUnmapMemory((*device)(), stagingBufferMemory);

		std::tie(vertexBuffer, vertexBufferMemory) = initBuffer(
			bufferSize,
			VK_BUFFER_USAGE_TRANSFER_DST_BIT | VK_BUFFER_USAGE_VERTEX_BUFFER_BIT,
			VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT
		);
		copyBuffer(stagingBuffer, vertexBuffer, bufferSize);
		vkDestroyBuffer((*device)(), stagingBuffer, nullptr);
		vkFreeMemory((*device)(), stagingBufferMemory, nullptr);
	}

	void initIndexBuffer() {
		VkDeviceSize bufferSize = sizeof(indices[0]) * indices.size();

		VkBuffer stagingBuffer;
		VkDeviceMemory stagingBufferMemory;
		std::tie(stagingBuffer, stagingBufferMemory) = initBuffer(
			bufferSize,
			VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
			VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT
		);

		void * data = {};
		vkMapMemory(
			(*device)(),
			stagingBufferMemory,
			0,
			bufferSize,
			0,
			&data
		);
		memcpy(data, indices.data(), (size_t)bufferSize);
		vkUnmapMemory((*device)(), stagingBufferMemory);

		std::tie(indexBuffer, indexBufferMemory) = initBuffer(
			bufferSize,
			VK_BUFFER_USAGE_TRANSFER_DST_BIT | VK_BUFFER_USAGE_INDEX_BUFFER_BIT,
			VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT
		);
		copyBuffer(stagingBuffer, indexBuffer, bufferSize);
		vkDestroyBuffer((*device)(), stagingBuffer, nullptr);
		vkFreeMemory((*device)(), stagingBufferMemory, nullptr);
	}

	void initUniformBuffers() {
		VkDeviceSize bufferSize = sizeof(UniformBufferObject);

		uniformBuffers.resize(MAX_FRAMES_IN_FLIGHT);
		uniformBuffersMemory.resize(MAX_FRAMES_IN_FLIGHT);
		uniformBuffersMapped.resize(MAX_FRAMES_IN_FLIGHT);

		for (size_t i = 0; i < MAX_FRAMES_IN_FLIGHT; i++) {
			std::tie(uniformBuffers[i], uniformBuffersMemory[i]) = initBuffer(
				bufferSize,
				VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT,
				VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT
			);
			
			vkMapMemory(
				(*device)(),
				uniformBuffersMemory[i],
				0,
				bufferSize,
				0,
				&uniformBuffersMapped[i]
			);
		}
	}

	std::pair<VkBuffer, VkDeviceMemory> initBuffer(
		VkDeviceSize size,
		VkBufferUsageFlags usage,
		VkMemoryPropertyFlags properties
	) {
		VkBuffer buffer;
		VkDeviceMemory bufferMemory;
		
		VkBufferCreateInfo bufferInfo = {};
		bufferInfo.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
		bufferInfo.size = size;
		bufferInfo.usage = usage;
		bufferInfo.sharingMode = VK_SHARING_MODE_EXCLUSIVE;

		VULKAN_SAFE(vkCreateBuffer, (*device)(), &bufferInfo, nullptr, &buffer);

		VkMemoryRequirements memRequirements;
		vkGetBufferMemoryRequirements((*device)(), buffer, &memRequirements);

		VkMemoryAllocateInfo allocInfo = {};
		allocInfo.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
		allocInfo.allocationSize = memRequirements.size;
		allocInfo.memoryTypeIndex = findMemoryType(memRequirements.memoryTypeBits, properties);

		VULKAN_SAFE(vkAllocateMemory, (*device)(), &allocInfo, nullptr, &bufferMemory);

		vkBindBufferMemory((*device)(), buffer, bufferMemory, 0);
	
		return std::make_pair(buffer, bufferMemory);
	}

	void copyBuffer(VkBuffer srcBuffer, VkBuffer dstBuffer, VkDeviceSize size) {
		VkCommandBufferAllocateInfo allocInfo = {};
		allocInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
		allocInfo.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
		allocInfo.commandPool = commandPool;
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

		vkFreeCommandBuffers((*device)(), commandPool, 1, &commandBuffer);
	}

	uint32_t findMemoryType(uint32_t typeFilter, VkMemoryPropertyFlags properties) {
		VkPhysicalDeviceMemoryProperties memProperties;
		vkGetPhysicalDeviceMemoryProperties((*physicalDevice)(), &memProperties);

		for (uint32_t i = 0; i < memProperties.memoryTypeCount; i++) {
			if ((typeFilter & (1 << i)) && (memProperties.memoryTypes[i].propertyFlags & properties) == properties) {
				return i;
			}
		}

		throw Common::Exception() << ("failed to find suitable memory type!");
	}


	void initCommandBuffers() {
		commandBuffers.resize(MAX_FRAMES_IN_FLIGHT);
		VkCommandBufferAllocateInfo allocInfo = {};
		allocInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
		allocInfo.commandPool = commandPool;
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
			vkCmdBindPipeline(commandBuffer, VK_PIPELINE_BIND_POINT_GRAPHICS, graphicsPipeline);

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

			VkBuffer vertexBuffers[] = {vertexBuffer};
			VkDeviceSize offsets[] = {0};
			vkCmdBindVertexBuffers(commandBuffer, 0, 1, vertexBuffers, offsets);

			vkCmdBindIndexBuffer(commandBuffer, indexBuffer, 0, VK_INDEX_TYPE_UINT16);

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

	decltype(std::chrono::high_resolution_clock::now()) startTime = std::chrono::high_resolution_clock::now();
	
	void updateUniformBuffer(uint32_t currentFrame_) {
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
