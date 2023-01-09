#include "SDLApp/SDLApp.h"
#include "Common/Exception.h"
#include "Common/File.h"
#include <SDL_vulkan.h>
#include <vulkan/vulkan.hpp>
#include <iostream>	//debugging only
#include <set>

// why do I think there are already similar classes in vulkan.hpp?

struct VulkanInstance {
protected:	
	VkInstance instance = {};
public:	
	decltype(instance) operator()() const { return instance; }

	~VulkanInstance() {
		if (instance) vkDestroyInstance(instance, nullptr);
	}
	
	VulkanInstance(::SDLApp::SDLApp const * const app, bool enableValidationLayers) {
		// vkCreateInstance needs appInfo
		
		VkApplicationInfo appInfo = {};
		appInfo.sType = VK_STRUCTURE_TYPE_APPLICATION_INFO;
		auto title = app->getTitle();
		appInfo.pApplicationName = title.c_str();
		appInfo.applicationVersion = VK_MAKE_VERSION(1, 0, 0);
		appInfo.pEngineName = "No Engine";
		appInfo.engineVersion = VK_MAKE_VERSION(1, 0, 0);
		appInfo.apiVersion = VK_API_VERSION_1_0;

		// debug output

		{
			std::vector<VkLayerProperties> availableLayers;
			uint32_t layerCount = {};
			vkEnumerateInstanceLayerProperties(&layerCount, nullptr);
			availableLayers.resize(layerCount);
			vkEnumerateInstanceLayerProperties(&layerCount, availableLayers.data());
			std::cout << "vulkan layers:" << std::endl;
			for (auto const & layer : availableLayers) {
				std::cout << "\t" << layer.layerName << std::endl;
			}
		}
		
		// vkCreateInstance needs layerNames

		std::vector<const char *> layerNames;
		if (enableValidationLayers) {
			//insert which of those into our layerName for creating something or something
			//layerNames.push_back("VK_LAYER_LUNARG_standard_validation");	//nope
			layerNames.push_back("VK_LAYER_KHRONOS_validation");	//nope
		}
		
		// vkCreateInstance needs extensions

		VkInstanceCreateInfo createInfo = {};
		createInfo.sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO;
		createInfo.pApplicationInfo = &appInfo;
		createInfo.enabledLayerCount = layerNames.size();
		createInfo.ppEnabledLayerNames = layerNames.data();
		
		auto extensions = getRequiredExtensions(app, enableValidationLayers);
		createInfo.enabledExtensionCount = extensions.size();
		createInfo.ppEnabledExtensionNames = extensions.data();
		{
			VkResult res = vkCreateInstance(&createInfo, nullptr, &instance);
			if (res != VK_SUCCESS) {
				throw Common::Exception() << "vkCreateInstance failed: " << res;
			}
		}
	}
protected:
	std::vector<char const *> getRequiredExtensions(::SDLApp::SDLApp const * const app, bool enableValidationLayers) {
		uint32_t extensionCount = {};
		if (SDL_Vulkan_GetInstanceExtensions(app->getWindow(), &extensionCount, nullptr) == SDL_FALSE) {
			throw Common::Exception() << "SDL_Vulkan_GetInstanceExtensions failed: " << SDL_GetError();
		}
		
		std::vector<const char *> extensions(extensionCount);
		if (SDL_Vulkan_GetInstanceExtensions(app->getWindow(), &extensionCount, extensions.data()) == SDL_FALSE) {
			throw Common::Exception() << "SDL_Vulkan_GetInstanceExtensions failed: " << SDL_GetError();
		}

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
	VkDebugUtilsMessengerEXT debugMessenger = {};
	VkInstance instance;	//from VulkanCommon, needs to be held for dtor to work
	
	// TODO reflection, pair, name, and constexpr lambda iterator across these to load them all at once
	// then put it in another class so other classes can access it and not just this class.
	PFN_vkCreateDebugUtilsMessengerEXT vkCreateDebugUtilsMessengerEXT = {};
	PFN_vkDestroyDebugUtilsMessengerEXT vkDestroyDebugUtilsMessengerEXT = {};

public:
	~VulkanDebugMessenger() {
		// call destroy function
		if (vkDestroyDebugUtilsMessengerEXT && debugMessenger) {
			vkDestroyDebugUtilsMessengerEXT(instance, debugMessenger, nullptr);
		}
	}

	VulkanDebugMessenger(
		VkInstance instance_
	) : instance(instance_) {
		// get ext func ptrs

		vkCreateDebugUtilsMessengerEXT = (PFN_vkCreateDebugUtilsMessengerEXT)vkGetInstanceProcAddr(instance, "vkCreateDebugUtilsMessengerEXT");
		if (!vkCreateDebugUtilsMessengerEXT) {
			throw Common::Exception() << "vkGetInstanceProcAddr vkCreateDebugUtilsMessengerEXT failed";
		}
		vkDestroyDebugUtilsMessengerEXT = (PFN_vkDestroyDebugUtilsMessengerEXT)vkGetInstanceProcAddr(instance, "vkDestroyDebugUtilsMessengerEXT");
		if (!vkDestroyDebugUtilsMessengerEXT) {
			throw Common::Exception() << "vkGetInstanceProcAddr vkDestroyDebugUtilsMessengerEXT failed";
		}

		// call create function
		
		VkDebugUtilsMessengerCreateInfoEXT createInfo = {};
		createInfo.sType = VK_STRUCTURE_TYPE_DEBUG_UTILS_MESSENGER_CREATE_INFO_EXT;
		createInfo.messageSeverity = VK_DEBUG_UTILS_MESSAGE_SEVERITY_VERBOSE_BIT_EXT | VK_DEBUG_UTILS_MESSAGE_SEVERITY_WARNING_BIT_EXT | VK_DEBUG_UTILS_MESSAGE_SEVERITY_ERROR_BIT_EXT;
		createInfo.messageType = VK_DEBUG_UTILS_MESSAGE_TYPE_GENERAL_BIT_EXT | VK_DEBUG_UTILS_MESSAGE_TYPE_VALIDATION_BIT_EXT | VK_DEBUG_UTILS_MESSAGE_TYPE_PERFORMANCE_BIT_EXT;
		createInfo.pfnUserCallback = debugCallback;

		if (vkCreateDebugUtilsMessengerEXT(instance, &createInfo, nullptr, &debugMessenger) != VK_SUCCESS) {
			throw Common::Exception() << "vkCreateDebugUtilsMessengerEXT  failed!";
		}
	}

protected:
	static VKAPI_ATTR VkBool32 VKAPI_CALL debugCallback(VkDebugUtilsMessageSeverityFlagBitsEXT messageSeverity, VkDebugUtilsMessageTypeFlagsEXT messageType, const VkDebugUtilsMessengerCallbackDataEXT* pCallbackData, void* pUserData) {
		std::cerr << "validation layer: " << pCallbackData->pMessage << std::endl;

		return VK_FALSE;
	}
};

struct VulkanSurface {
protected:	
	VkSurfaceKHR surface;
	VkInstance instance;	//from VulkanCommon, needs to be held for dtor to work
public:
	decltype(surface) operator()() const { return surface; }
	
	~VulkanSurface() {
		if (surface) vkDestroySurfaceKHR(instance, surface, nullptr);
	}

	VulkanSurface(
		::SDLApp::SDLApp const * const app,
		VkInstance instance_
	) : instance(instance_) {
		// https://vulkan-tutorial.com/Drawing_a_triangle/Presentation/Window_surface
		if (!SDL_Vulkan_CreateSurface(app->getWindow(), instance, &surface)) {
			 throw Common::Exception() << "vkCreateWaylandSurfaceKHR failed: " << SDL_GetError();
		}
	}
};

struct VulkanPhysicalDevice {
protected:	
	VkPhysicalDevice physicalDevice = {};
public:
	decltype(physicalDevice) operator()() const { return physicalDevice; }

	~VulkanPhysicalDevice() {}
	
	VulkanPhysicalDevice(
		VkInstance instance,
		VkSurfaceKHR surface,								// needed by isDeviceSuitable -> findQueueFamilie
		std::vector<char const *> const & deviceExtensions	// needed by isDeviceSuitable -> checkDeviceExtensionSupport
	) {
		uint32_t deviceCount = {};
		vkEnumeratePhysicalDevices(instance, &deviceCount, nullptr);
		if (!deviceCount) {
			throw Common::Exception() << "failed to find GPUs with Vulkan support!";
		}
		std::vector<VkPhysicalDevice> devices(deviceCount);
		vkEnumeratePhysicalDevices(instance, &deviceCount, devices.data());
		
		std::cout << "devices:" << std::endl;
		for (auto const & device : devices) {
			VkPhysicalDeviceProperties deviceProperties;
			vkGetPhysicalDeviceProperties(device, &deviceProperties);
			std::cout 
				<< "\t"
				<< deviceProperties.deviceName 
				<< " type=" << deviceProperties.deviceType
				<< std::endl;

			if (isDeviceSuitable(device, surface, deviceExtensions)) {
				physicalDevice = device;
				break;
			}
		}

		if (!physicalDevice) {
			throw Common::Exception() << "failed to find a suitable GPU!";
		}
	}

protected:
	static bool isDeviceSuitable(
		VkPhysicalDevice physicalDevice,
		VkSurfaceKHR surface,								// needed by findQueueFamilies, querySwapChainSupport
		std::vector<char const *> const & deviceExtensions	// needed by checkDeviceExtensionSupport
	) {

#if 0	// i'm not seeing queue families indices and the actual physicalDevice info query overlap
		// or is querying individual devices properties not a thing anymore?
		// do you just search for the queue family bit?  graphics? compute? whatever?

		VkPhysicalDeviceProperties deviceProperties;
		vkGetPhysicalDeviceProperties(physicalDevice, &deviceProperties);
		VkPhysicalDeviceFeatures deviceFeatures;
		vkGetPhysicalDeviceFeatures(physicalDevice, &deviceFeatures);
		// TODO sort by score and pick the best
		return deviceProperties.deviceType == VK_PHYSICAL_DEVICE_TYPE_DISCRETE_GPU
			|| deviceProperties.deviceType == VK_PHYSICAL_DEVICE_TYPE_INTEGRATED_GPU
			|| deviceProperties.deviceType == VK_PHYSICAL_DEVICE_TYPE_VIRTUAL_GPU
		;
			// && deviceFeatures.geometryShader;
#endif

		auto indices = findQueueFamilies(physicalDevice, surface);
		
		bool extensionsSupported = checkDeviceExtensionSupport(physicalDevice, deviceExtensions);

		bool swapChainAdequate = false;
		if (extensionsSupported) {
			auto swapChainSupport = querySwapChainSupport(physicalDevice, surface);
			swapChainAdequate = !swapChainSupport.formats.empty() && !swapChainSupport.presentModes.empty();
		}

		return indices.isComplete() 
			&& extensionsSupported 
			&& swapChainAdequate;
	}

	//used by isDeviceSuitable
	static bool checkDeviceExtensionSupport(
		VkPhysicalDevice physicalDevice,
		std::vector<char const *> const & deviceExtensions
	) {
		uint32_t extensionCount;
		vkEnumerateDeviceExtensionProperties(physicalDevice, nullptr, &extensionCount, nullptr);

		std::vector<VkExtensionProperties> availableExtensions(extensionCount);
		vkEnumerateDeviceExtensionProperties(physicalDevice, nullptr, &extensionCount, availableExtensions.data());

		std::set<std::string> requiredExtensions(deviceExtensions.begin(), deviceExtensions.end());

		for (const auto& extension : availableExtensions) {
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
	static QueueFamilyIndices findQueueFamilies(
		VkPhysicalDevice physicalDevice,
		VkSurfaceKHR surface
	) {
		QueueFamilyIndices indices;

		uint32_t queueFamilyCount = 0;
		vkGetPhysicalDeviceQueueFamilyProperties(physicalDevice, &queueFamilyCount, nullptr);
		std::vector<VkQueueFamilyProperties> queueFamilies(queueFamilyCount);
		vkGetPhysicalDeviceQueueFamilyProperties(physicalDevice, &queueFamilyCount, queueFamilies.data());


		for (uint32_t i = 0; i < queueFamilies.size(); ++i) {
			auto const & f = queueFamilies[i];
			if (f.queueFlags & VK_QUEUE_GRAPHICS_BIT) {
				indices.graphicsFamily = i;
			}
	
			VkBool32 presentSupport = false;
			vkGetPhysicalDeviceSurfaceSupportKHR(physicalDevice, i, surface, &presentSupport);
			if (presentSupport) {
				indices.presentFamily = i;
			}
			
			if (indices.isComplete()) return indices;
		}

		throw Common::Exception() << "couldn't find all indices";
	}

public:
	struct SwapChainSupportDetails {
		VkSurfaceCapabilitiesKHR capabilities;
		std::vector<VkSurfaceFormatKHR> formats;
		std::vector<VkPresentModeKHR> presentModes;
	};

	static SwapChainSupportDetails querySwapChainSupport(
		VkPhysicalDevice physicalDevice,
		VkSurfaceKHR surface
	) {
		SwapChainSupportDetails details;

		vkGetPhysicalDeviceSurfaceCapabilitiesKHR(physicalDevice, surface, &details.capabilities);

		uint32_t formatCount;
		vkGetPhysicalDeviceSurfaceFormatsKHR(physicalDevice, surface, &formatCount, nullptr);

		if (formatCount != 0) {
			details.formats.resize(formatCount);
			vkGetPhysicalDeviceSurfaceFormatsKHR(physicalDevice, surface, &formatCount, details.formats.data());
		}

		uint32_t presentModeCount;
		vkGetPhysicalDeviceSurfacePresentModesKHR(physicalDevice, surface, &presentModeCount, nullptr);

		if (presentModeCount != 0) {
			details.presentModes.resize(presentModeCount);
			vkGetPhysicalDeviceSurfacePresentModesKHR(physicalDevice, surface, &presentModeCount, details.presentModes.data());
		}

		return details;
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

	// used by 
	//	VulkanPhysicalDevice::checkDeviceExtensionSupport
	//	initLogicalDevice
	std::vector<char const *> const deviceExtensions = {
		VK_KHR_SWAPCHAIN_EXTENSION_NAME
	};


	VulkanCommon(::SDLApp::SDLApp const * const app_) 
	: app(app_) {
		// TODO half tempting to put this inside debug init instead of here
		if (enableValidationLayers && !checkValidationLayerSupport()) {
			throw Common::Exception() << "validation layers requested, but not available!";
		}

		// hmm, maybe instance should be a shared_ptr and then passed to debug, surface, and physicalDevice ?
		instance = std::make_unique<VulkanInstance>(app, enableValidationLayers);
		
		if (enableValidationLayers) {
			debug = std::make_unique<VulkanDebugMessenger>((*instance)());
		}
		
		surface = std::make_unique<VulkanSurface>(app, (*instance)());
		
		{
			// used in other inits  ... initLogicalDevice and initSwapChain
			// so we don't need to store this as a member, but only a scoped var for the duration of the ctor
			auto physicalDevice = std::make_unique<VulkanPhysicalDevice>((*instance)(), (*surface)(), deviceExtensions);
			initLogicalDevice(physicalDevice.get());
			initSwapChain(physicalDevice.get());
		}
		initImageView();
		initGraphicsPipeline();
	}

	~VulkanCommon() {
		for (auto imageView : swapChainImageViews) {
			vkDestroyImageView(device, imageView, nullptr);
		}
		if (swapChain) vkDestroySwapchainKHR(device, swapChain, nullptr);
		if (device) vkDestroyDevice(device, nullptr);
		surface = nullptr;
		debug = nullptr;
		instance = nullptr;
	}

	// validationLayers matches in checkValidationLayerSupport and initLogicalDevice
	std::vector<char const *> const validationLayers = {
		"VK_LAYER_KHRONOS_validation"
	};

	bool checkValidationLayerSupport() {
		uint32_t layerCount;
		vkEnumerateInstanceLayerProperties(&layerCount, nullptr);

		std::vector<VkLayerProperties> availableLayers(layerCount);
		vkEnumerateInstanceLayerProperties(&layerCount, availableLayers.data());

		for (char const * const layerName : validationLayers) {
			bool layerFound = false;
			for (auto const & layerProperties : availableLayers) {
				if (strcmp(layerName, layerProperties.layerName) == 0) {
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

	VkDevice device = {};
	VkQueue graphicsQueue;
	VkQueue presentQueue;

	virtual void initLogicalDevice(VulkanPhysicalDevice * const physicalDevice) {
		auto indices = VulkanPhysicalDevice::findQueueFamilies((*physicalDevice)(), (*surface)());

		std::vector<VkDeviceQueueCreateInfo> queueCreateInfos;
		std::set<uint32_t> uniqueQueueFamilies = {
			indices.graphicsFamily.value(),
			indices.presentFamily.value(),
		};

		float queuePriority = 1.0f;
		for (uint32_t queueFamily : uniqueQueueFamilies) {
			VkDeviceQueueCreateInfo queueCreateInfo = {};
			queueCreateInfo.sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO;
			queueCreateInfo.queueFamilyIndex = queueFamily;
			queueCreateInfo.queueCount = 1;
			queueCreateInfo.pQueuePriorities = &queuePriority;
			queueCreateInfos.push_back(queueCreateInfo);
		}

		VkDeviceCreateInfo createInfo = {};
		createInfo.sType = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO;
		
		createInfo.queueCreateInfoCount = static_cast<uint32_t>(queueCreateInfos.size());
		createInfo.pQueueCreateInfos = queueCreateInfos.data();
		
		VkPhysicalDeviceFeatures deviceFeatures = {}; // empty
		createInfo.pEnabledFeatures = &deviceFeatures;
		
		createInfo.enabledExtensionCount = static_cast<uint32_t>(deviceExtensions.size());
		createInfo.ppEnabledExtensionNames = deviceExtensions.data();
		
		if (enableValidationLayers) {
			createInfo.enabledLayerCount = static_cast<uint32_t>(validationLayers.size());
			createInfo.ppEnabledLayerNames = validationLayers.data();
		} else {
			createInfo.enabledLayerCount = 0;
		}
		{
			VkResult res = vkCreateDevice((*physicalDevice)(), &createInfo, nullptr, &device);
			if (res != VK_SUCCESS) throw Common::Exception() << "vkCreateDevice failed: " << res;
		}
	
		vkGetDeviceQueue(device, indices.graphicsFamily.value(), 0, &graphicsQueue);
		vkGetDeviceQueue(device, indices.presentFamily.value(), 0, &presentQueue);
	}

	VkSwapchainKHR swapChain;
	std::vector<VkImage> swapChainImages;
	VkFormat swapChainImageFormat;
	VkExtent2D swapChainExtent;
	std::vector<VkImageView> swapChainImageViews;
		
	void initSwapChain(VulkanPhysicalDevice * const physicalDevice) {
		auto swapChainSupport = VulkanPhysicalDevice::querySwapChainSupport((*physicalDevice)(), (*surface)());

		VkSurfaceFormatKHR surfaceFormat = chooseSwapSurfaceFormat(swapChainSupport.formats);
		VkPresentModeKHR presentMode = chooseSwapPresentMode(swapChainSupport.presentModes);
		VkExtent2D extent = chooseSwapExtent(swapChainSupport.capabilities);

		uint32_t imageCount = swapChainSupport.capabilities.minImageCount + 1;
		if (swapChainSupport.capabilities.maxImageCount > 0 && imageCount > swapChainSupport.capabilities.maxImageCount) {
			imageCount = swapChainSupport.capabilities.maxImageCount;
		}

		VkSwapchainCreateInfoKHR createInfo{};
		createInfo.sType = VK_STRUCTURE_TYPE_SWAPCHAIN_CREATE_INFO_KHR;
		createInfo.surface = (*surface)();

		createInfo.minImageCount = imageCount;
		createInfo.imageFormat = surfaceFormat.format;
		createInfo.imageColorSpace = surfaceFormat.colorSpace;
		createInfo.imageExtent = extent;
		createInfo.imageArrayLayers = 1;
		createInfo.imageUsage = VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT;

		auto indices = VulkanPhysicalDevice::findQueueFamilies((*physicalDevice)(), (*surface)());
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

		createInfo.oldSwapchain = VK_NULL_HANDLE;

		if (vkCreateSwapchainKHR(device, &createInfo, nullptr, &swapChain) != VK_SUCCESS) {
			throw Common::Exception() << "failed to create swap chain!";
		}

		vkGetSwapchainImagesKHR(device, swapChain, &imageCount, nullptr);
		swapChainImages.resize(imageCount);
		vkGetSwapchainImagesKHR(device, swapChain, &imageCount, swapChainImages.data());

		swapChainImageFormat = surfaceFormat.format;
		swapChainExtent = extent;
	}
		
	VkSurfaceFormatKHR chooseSwapSurfaceFormat(const std::vector<VkSurfaceFormatKHR>& availableFormats) {
		for (const auto& availableFormat : availableFormats) {
			if (availableFormat.format == VK_FORMAT_B8G8R8A8_SRGB && availableFormat.colorSpace == VK_COLOR_SPACE_SRGB_NONLINEAR_KHR) {
				return availableFormat;
			}
		}

		return availableFormats[0];
	}
	
	VkPresentModeKHR chooseSwapPresentMode(const std::vector<VkPresentModeKHR>& availablePresentModes) {
		for (const auto& availablePresentMode : availablePresentModes) {
			if (availablePresentMode == VK_PRESENT_MODE_MAILBOX_KHR) {
				return availablePresentMode;
			}
		}

		return VK_PRESENT_MODE_FIFO_KHR;
	}

	VkExtent2D chooseSwapExtent(const VkSurfaceCapabilitiesKHR& capabilities) {
		if (capabilities.currentExtent.width != std::numeric_limits<uint32_t>::max()) {
			return capabilities.currentExtent;
		} else {
			VkExtent2D actualExtent = {
				static_cast<uint32_t>(app->getScreenSize().x),
				static_cast<uint32_t>(app->getScreenSize().y)
			};

			actualExtent.width = std::clamp(actualExtent.width, capabilities.minImageExtent.width, capabilities.maxImageExtent.width);
			actualExtent.height = std::clamp(actualExtent.height, capabilities.minImageExtent.height, capabilities.maxImageExtent.height);

			return actualExtent;
		}
	}
	
	void initImageView() {
		swapChainImageViews.resize(swapChainImages.size());

		for (size_t i = 0; i < swapChainImages.size(); i++) {
			VkImageViewCreateInfo createInfo = {};
			createInfo.sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO;
			createInfo.image = swapChainImages[i];
			createInfo.viewType = VK_IMAGE_VIEW_TYPE_2D;
			createInfo.format = swapChainImageFormat;
			createInfo.components.r = VK_COMPONENT_SWIZZLE_IDENTITY;
			createInfo.components.g = VK_COMPONENT_SWIZZLE_IDENTITY;
			createInfo.components.b = VK_COMPONENT_SWIZZLE_IDENTITY;
			createInfo.components.a = VK_COMPONENT_SWIZZLE_IDENTITY;
			createInfo.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
			createInfo.subresourceRange.baseMipLevel = 0;
			createInfo.subresourceRange.levelCount = 1;
			createInfo.subresourceRange.baseArrayLayer = 0;
			createInfo.subresourceRange.layerCount = 1;

			VkResult res = vkCreateImageView(device, &createInfo, nullptr, &swapChainImageViews[i]);
			if (res != VK_SUCCESS) {
				throw Common::Exception() << "vkCreateImageView failed: " << res;
			}
		}
	}
		
	void initGraphicsPipeline() {
	    auto vertShaderCode = Common::File::read("shader-vert.spv");
        auto fragShaderCode = Common::File::read("shader-frag.spv");

        VkShaderModule vertShaderModule = createShaderModule(vertShaderCode);
        VkShaderModule fragShaderModule = createShaderModule(fragShaderCode);

        VkPipelineShaderStageCreateInfo vertShaderStageInfo{};
        vertShaderStageInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
        vertShaderStageInfo.stage = VK_SHADER_STAGE_VERTEX_BIT;
        vertShaderStageInfo.module = vertShaderModule;
        // GLSL uses 'main', but clspv doesn't allow 'main', so ....
		//vertShaderStageInfo.pName = "main";
		vertShaderStageInfo.pName = "vert";

        VkPipelineShaderStageCreateInfo fragShaderStageInfo{};
        fragShaderStageInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
        fragShaderStageInfo.stage = VK_SHADER_STAGE_FRAGMENT_BIT;
        fragShaderStageInfo.module = fragShaderModule;
        //fragShaderStageInfo.pName = "main";
        fragShaderStageInfo.pName = "frag";

        VkPipelineShaderStageCreateInfo shaderStages[] = {
			vertShaderStageInfo,
			fragShaderStageInfo,
		};

        vkDestroyShaderModule(device, fragShaderModule, nullptr);
        vkDestroyShaderModule(device, vertShaderModule, nullptr);

	}

    VkShaderModule createShaderModule(std::string const & code) {
        VkShaderModuleCreateInfo createInfo{};
        createInfo.sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO;
        createInfo.codeSize = code.length();
        createInfo.pCode = reinterpret_cast<const uint32_t*>(code.data());

        VkShaderModule shaderModule;
        auto res = vkCreateShaderModule(device, &createInfo, nullptr, &shaderModule);
		if (res != VK_SUCCESS) throw Common::Exception() << "vkCreateShaderModule failed: " << res;

        return shaderModule;
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

	std::string getTitle() {
		return "Vulkan Test";
	}
	
	virtual Uint32 getSDLCreateWindowFlags() {
		return Super::getSDLCreateWindowFlags() | SDL_WINDOW_VULKAN;
	}
};

SDLAPP_MAIN(Test)
