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

#define SDL_VULKAN_SAFE(f, ...) {\
	if (f(__VA_ARGS__) == SDL_FALSE) {\
		throw Common::Exception() << FILE_AND_LINE " " #f " failed: " << SDL_GetError();\
	}\
}

template<typename real>
real degToRad(real x) {
	return x * (real)(M_PI / 180.);
}

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
		return vk::VertexInputBindingDescription()
			.setBinding(0)
			.setStride(sizeof(Vertex))
			.setInputRate(vk::VertexInputRate::eVertex);
	}

	static auto getAttributeDescriptions() {
		return Common::make_array(
			vk::VertexInputAttributeDescription()
				.setLocation(0)
				.setBinding(0)
				.setFormat(vk::Format::eR32G32B32Sfloat)
				.setOffset(offsetof(Vertex, pos)),
			vk::VertexInputAttributeDescription()
				.setLocation(1)
				.setBinding(0)
				.setFormat(vk::Format::eR32G32B32Sfloat)
				.setOffset(offsetof(Vertex, color)),
			vk::VertexInputAttributeDescription()
				.setLocation(2)
				.setBinding(0)
				.setFormat(vk::Format::eR32G32Sfloat)
				.setOffset(offsetof(Vertex, texCoord))
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


struct VulkanInstance {

	// this does result in vkCreateInstance,
	//  but the way it gest there is very application-specific
	static auto create(
		auto const & ctx,
		::SDLApp::SDLApp const * const app,
		bool const enableValidationLayers
	) {
		// debug output
		std::cout << "vulkan layers:" << std::endl;
		for (auto const & layer : vk::enumerateInstanceLayerProperties()) {
			std::cout << "\t" << layer.layerName.data() << std::endl;
		}

		// vk::ApplicationInfo needs title:
		auto title = app->getTitle();
		
		// vkCreateInstance needs appInfo
		auto appInfo = vk::ApplicationInfo()
			.setPApplicationName(title.c_str())
			.setApplicationVersion(VK_MAKE_VERSION(1, 0, 0))
			.setPEngineName("No Engine")
			.setEngineVersion(VK_MAKE_VERSION(1, 0, 0))
			.setApiVersion(VK_API_VERSION_1_0);

		// vkCreateInstance needs layerNames
		std::vector<char const *> layerNames;
		if (enableValidationLayers) {
			//insert which of those into our layerName for creating something or something
			//layerNames.push_back("VK_LAYER_LUNARG_standard_validation");	//nope
			layerNames.push_back("VK_LAYER_KHRONOS_validation");	//nope
		}
		
		// vkCreateInstance needs extensions
		auto extensions = getRequiredExtensions(app, enableValidationLayers);

		return vk::raii::Instance(
			ctx,
			vk::InstanceCreateInfo()
				.setPApplicationInfo(&appInfo)
				.setPEnabledLayerNames(layerNames)
				.setPEnabledExtensionNames(extensions)
		);
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

struct VulkanPhysicalDevice {
	// used by the application for specific physical device querying (should be a subclass of the general vk::PhysicalDevice)
	static auto create(
		vk::raii::Instance const & instance,
		vk::raii::SurfaceKHR const & surface,
		std::vector<char const *> const & deviceExtensions
	) {
		auto physDevs = instance.enumeratePhysicalDevices();
		//debug:
		std::cout << "devices:" << std::endl;
		for (auto const & physDev : physDevs) {
			auto props = physDev.getProperties();
			std::cout
				<< "\t"
				<< props.deviceName.data()
				<< " type=" << (uint32_t)props.deviceType
				<< std::endl;
		}

		for (auto const & physDev : physDevs) {
			if (isDeviceSuitable(physDev, surface, deviceExtensions)) {
				// I can return a raii::PHysicalDevice as-is here
				// buut the source variable can't be initialized with a default/empty cotr
				// so i'll make the source a uique_ptr
				// but then that means i have to reutnr a make_unique....... 
				// hmm.......
				return physDev;
			}
		}

		throw Common::Exception() << "failed to find a suitable GPU!";
	}

public:
	struct SwapChainSupportDetails {
		vk::SurfaceCapabilitiesKHR capabilities;
		std::vector<vk::SurfaceFormatKHR> formats;
		std::vector<vk::PresentModeKHR> presentModes;
	};

	static auto querySwapChainSupport(
		vk::raii::PhysicalDevice const & physDev,
		vk::raii::SurfaceKHR const & surface
	) {
		return SwapChainSupportDetails{
			.capabilities = physDev.getSurfaceCapabilitiesKHR(*surface),
			.formats = physDev.getSurfaceFormatsKHR(*surface),
			.presentModes = physDev.getSurfacePresentModesKHR(*surface)
		};
	}

protected:
	static bool isDeviceSuitable(
		vk::raii::PhysicalDevice const & physDev,
		vk::raii::SurfaceKHR const & surface,
		std::vector<char const *> const & deviceExtensions
	) {
		auto indices = findQueueFamilies(physDev, surface);
		bool extensionsSupported = checkDeviceExtensionSupport(physDev, deviceExtensions);
		bool swapChainAdequate = false;
		if (extensionsSupported) {
			auto swapChainSupport = querySwapChainSupport(physDev, surface);
			swapChainAdequate = !swapChainSupport.formats.empty() && !swapChainSupport.presentModes.empty();
		}
		vk::PhysicalDeviceFeatures features = physDev.getFeatures();
		return indices.isComplete()
			&& extensionsSupported
			&& swapChainAdequate
			&& features.samplerAnisotropy;
	}

	static bool checkDeviceExtensionSupport(
		vk::raii::PhysicalDevice const & physDev,
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
		vk::raii::PhysicalDevice const & physDev
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
		vk::raii::PhysicalDevice const & physDev,
		vk::raii::SurfaceKHR const & surface
	) {
		QueueFamilyIndices indices;
		auto queueFamilies = physDev.getQueueFamilyProperties();
		for (uint32_t i = 0; i < queueFamilies.size(); ++i) {
			auto const & f = queueFamilies[i];
			if (f.queueFlags & vk::QueueFlagBits::eGraphics) {
				indices.graphicsFamily = i;
			}
			if (physDev.getSurfaceSupportKHR(i, *surface)) {
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
		vk::raii::PhysicalDevice const & physDev
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
		vk::raii::PhysicalDevice const & physDev,
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
		vk::raii::PhysicalDevice const & physDev,
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

// separate class for sole purpose of constructing device
// I've lumped in 'graphicsQueue' and 'presentQueue' because their ctor depends on the 'indices' var which is used to ctor 'device' as well
struct VulkanDevice {
protected:	
	//owns:
	vk::raii::Device device;
	vk::raii::Queue graphicsQueue;
	vk::raii::Queue presentQueue;
public:
	auto const & operator()() const { return device; }
	auto const & getGraphicsQueue() const { return graphicsQueue; }
	auto const & getPresentQueue() const { return presentQueue; }

protected:
	static auto createDevice(
		vk::raii::PhysicalDevice const & physicalDevice,
		std::vector<char const *> const & deviceExtensions,
		bool enableValidationLayers,
		VulkanPhysicalDevice::QueueFamilyIndices const & indices
	) {
		// can't return the createInfo because it holds stack pointers
		// gotta create the device and then move it into the calling function
		auto queuePriorities = Common::make_array<float>(1);
		auto queueCreateInfos = std::vector<vk::DeviceQueueCreateInfo>{};
		for (uint32_t queueFamily : std::set<uint32_t>{
			indices.graphicsFamily.value(),
			indices.presentFamily.value(),
		}) {
			queueCreateInfos.push_back(
				vk::DeviceQueueCreateInfo()
					.setQueueFamilyIndex(queueFamily)
					.setQueuePriorities(queuePriorities)
			);
		}

		auto deviceFeatures = vk::PhysicalDeviceFeatures()
			.setSamplerAnisotropy(true);
		auto thisValidationLayers = std::vector<char const *>();
		if (enableValidationLayers) {
			thisValidationLayers = validationLayers;
		}	
		return vk::raii::Device(
			physicalDevice,
			vk::DeviceCreateInfo()
				.setQueueCreateInfos(queueCreateInfos)
				.setPEnabledLayerNames(thisValidationLayers)
				.setPEnabledExtensionNames(deviceExtensions)
				.setPEnabledFeatures(&deviceFeatures)
		);
	}
public:
	VulkanDevice(
		vk::raii::PhysicalDevice const & physicalDevice,
		std::vector<char const *> const & deviceExtensions,
		bool enableValidationLayers,
		VulkanPhysicalDevice::QueueFamilyIndices const & indices
	) : 
		device(createDevice(
			physicalDevice,
			deviceExtensions,
			enableValidationLayers,
			indices
		)),
		graphicsQueue(device, indices.graphicsFamily.value(), 0),
		presentQueue(device, indices.presentFamily.value(), 0)
	{}
};

struct VulkanRenderPass  {
	static auto create(
		vk::raii::PhysicalDevice const physicalDevice,
		vk::raii::Device const & device,
		vk::Format swapChainImageFormat,
		vk::SampleCountFlagBits msaaSamples
	) {
		auto attachments = Common::make_array(
			vk::AttachmentDescription()//colorAttachment
				.setFormat(swapChainImageFormat)
				.setSamples(msaaSamples)
				.setLoadOp(vk::AttachmentLoadOp::eClear)
				.setStoreOp(vk::AttachmentStoreOp::eStore)
				.setStencilLoadOp(vk::AttachmentLoadOp::eDontCare)
				.setStencilStoreOp(vk::AttachmentStoreOp::eDontCare)
				.setInitialLayout(vk::ImageLayout::eUndefined)
				.setFinalLayout(vk::ImageLayout::eColorAttachmentOptimal),
			vk::AttachmentDescription()//depthAttachment
				.setFormat(VulkanPhysicalDevice::findDepthFormat(physicalDevice))
				.setSamples(msaaSamples)
				.setLoadOp(vk::AttachmentLoadOp::eClear)
				.setStoreOp(vk::AttachmentStoreOp::eDontCare)
				.setStencilLoadOp(vk::AttachmentLoadOp::eDontCare)
				.setStencilStoreOp(vk::AttachmentStoreOp::eDontCare)
				.setInitialLayout(vk::ImageLayout::eUndefined)
				.setFinalLayout(vk::ImageLayout::eDepthStencilAttachmentOptimal),
			vk::AttachmentDescription()//colorAttachmentResolve
				.setFormat(swapChainImageFormat)
				.setSamples(vk::SampleCountFlagBits::e1)
				.setLoadOp(vk::AttachmentLoadOp::eDontCare)
				.setStoreOp(vk::AttachmentStoreOp::eStore)
				.setStencilLoadOp(vk::AttachmentLoadOp::eDontCare)
				.setStencilStoreOp(vk::AttachmentStoreOp::eDontCare)
				.setInitialLayout(vk::ImageLayout::eUndefined)
				.setFinalLayout(vk::ImageLayout::ePresentSrcKHR)
		);
		auto colorAttachmentRef = vk::AttachmentReference()
			.setAttachment(0)
			.setLayout(vk::ImageLayout::eColorAttachmentOptimal);
		auto depthAttachmentRef = vk::AttachmentReference()
			.setAttachment(1)
			.setLayout(vk::ImageLayout::eDepthStencilAttachmentOptimal);
		auto colorAttachmentResolveRef = vk::AttachmentReference()
			.setAttachment(2)
			.setLayout(vk::ImageLayout::eColorAttachmentOptimal);
		auto subpasses = Common::make_array(
			vk::SubpassDescription()
				.setPipelineBindPoint(vk::PipelineBindPoint::eGraphics)
				.setColorAttachmentCount(1)
				.setPColorAttachments(&colorAttachmentRef)
				.setPResolveAttachments(&colorAttachmentResolveRef)
				.setPDepthStencilAttachment(&depthAttachmentRef)
		);
		auto dependencies = Common::make_array(
			vk::SubpassDependency()
				.setSrcSubpass(VK_SUBPASS_EXTERNAL)
				.setDstSubpass(0)
				.setSrcStageMask(
					vk::PipelineStageFlagBits::eColorAttachmentOutput |
					vk::PipelineStageFlagBits::eEarlyFragmentTests
				)
				.setDstStageMask(
					vk::PipelineStageFlagBits::eColorAttachmentOutput |
					vk::PipelineStageFlagBits::eEarlyFragmentTests
				)
				.setSrcAccessMask({})
				.setDstAccessMask(
					vk::AccessFlagBits::eColorAttachmentWrite |
					vk::AccessFlagBits::eDepthStencilAttachmentWrite
				)
		);
		return vk::raii::RenderPass(
			device,
			vk::RenderPassCreateInfo()
				.setAttachments(attachments)
				.setSubpasses(subpasses)
				.setDependencies(dependencies)
		);
	}
};

struct VulkanSingleTimeCommand  {
protected:	
	//owns:
	std::vector<vk::raii::CommandBuffer> cmds;
	//held:
	vk::raii::Queue const & queue;
public:
	auto const & operator()() const { return cmds[0]; }

	VulkanSingleTimeCommand(
		vk::raii::Device const & device,
		vk::raii::Queue const & queue_,
		vk::raii::CommandPool const & commandPool
	) : 
		cmds(
			device.allocateCommandBuffers(
	//			commandPool,
				vk::CommandBufferAllocateInfo()
					.setCommandPool(*commandPool)
					.setLevel(vk::CommandBufferLevel::ePrimary)
					.setCommandBufferCount(1)
			)
		),
		queue(queue_)
	{
		// end part that matches
		// and this part kinda matches the start of 'recordCommandBuffer'
		cmds[0].begin(
			vk::CommandBufferBeginInfo()
				.setFlags(vk::CommandBufferUsageFlagBits::eOneTimeSubmit)
		);
		//end part that matches
	}
	
	~VulkanSingleTimeCommand() {
		cmds[0].end();
		auto vkcmds = Common::make_array(
			*cmds[0]
		);
		queue.submit(
			vk::SubmitInfo()
				.setCommandBuffers(vkcmds)
		);
		queue.waitIdle();
	}
};

struct VulkanCommandPool {
protected:
	//owns:
	vk::raii::CommandPool commandPool;
	//held:
	vk::raii::Device const & device;
	vk::raii::Queue const & graphicsQueue;
public:
	auto const & operator()() const { return commandPool; }
	
	VulkanCommandPool(
		vk::raii::Device const & device_,
		vk::raii::Queue const & graphicsQueue_,
		vk::CommandPoolCreateInfo const info
	) : commandPool(device_, info),
		device(device_),
		graphicsQueue(graphicsQueue_)
	{}

	//copies based on the graphicsQueue
	// used by makeBufferFromStaged
	void copyBuffer(
		vk::Buffer srcBuffer,	//staging vk::Buffer
		vk::Buffer dstBuffer,	//dest vk::Buffer
		vk::DeviceSize size
	) const {
		VulkanSingleTimeCommand(
			device,
			graphicsQueue,
			(*this)()
		)().copyBuffer(
			srcBuffer,
			dstBuffer,
			vk::BufferCopy()
			.setSize(size)
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
			vk::BufferImageCopy()
			.setImageSubresource(
				vk::ImageSubresourceLayers()
				.setAspectMask(vk::ImageAspectFlagBits::eColor)
				.setLayerCount(1)
			)
			.setImageExtent(vk::Extent3D(width, height, 1))
		);
	}

	void transitionImageLayout(
		vk::Image image,
		vk::ImageLayout oldLayout,
		vk::ImageLayout newLayout,
		uint32_t mipLevels
	) const {
		VulkanSingleTimeCommand commandBuffer(
			device,
			graphicsQueue,
			(*this)()
		);

		auto barrier = vk::ImageMemoryBarrier()
			.setOldLayout(oldLayout)
			.setNewLayout(newLayout)
			.setSrcQueueFamilyIndex(VK_QUEUE_FAMILY_IGNORED)
			.setDstQueueFamilyIndex(VK_QUEUE_FAMILY_IGNORED)
			.setImage(image)
			.setSubresourceRange(
				vk::ImageSubresourceRange()
					.setAspectMask(vk::ImageAspectFlagBits::eColor)
					.setLevelCount(mipLevels)
					.setLayerCount(1)
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

struct VulkanImageAndMemory {
protected:	
	vk::raii::Image image;
	vk::raii::DeviceMemory memory;
public:
	auto const & getImage() const { return image; }
	auto const & getMemory() const { return memory; }
	
	VulkanImageAndMemory(
		vk::raii::Image && image_,
		vk::raii::DeviceMemory && memory_
	) :	image(std::move(image_)),
		memory(std::move(memory_))
	{}

	VulkanImageAndMemory(VulkanImageAndMemory && o)
	: 	image(std::move(o.image)),
		memory(std::move(o.memory))
	{}

	VulkanImageAndMemory & operator=(VulkanImageAndMemory && o) {
		image = std::move(o.image);
		memory = std::move(o.memory);
		return *this;
	}
};

struct VulkanBufferMemoryAndMapped;
//copying the whole thing from VulkanImageAndMemory because I need the return type of operator= to be child-most, not parent-most
struct VulkanBufferAndMemory {
	friend struct VulkanBufferMemoryAndMapped;
protected:	
	vk::raii::Buffer buffer;
	vk::raii::DeviceMemory memory;
public:
	auto const & getBuffer() const { return buffer; }
	auto const & getMemory() const { return memory; }
	
	VulkanBufferAndMemory(
		vk::raii::Buffer && buffer_,
		vk::raii::DeviceMemory && memory_
	) :	buffer(std::move(buffer_)),
		memory(std::move(memory_))
	{}

	VulkanBufferAndMemory(VulkanBufferAndMemory && o)
	: 	buffer(std::move(o.buffer)),
		memory(std::move(o.memory))
	{}

	VulkanBufferAndMemory & operator=(VulkanBufferAndMemory && o) {
		buffer = std::move(o.buffer);
		memory = std::move(o.memory);
		return *this;
	}
};

struct VulkanBufferMemoryAndMapped {
protected:	
	std::unique_ptr<VulkanBufferAndMemory> bm;
	void * mapped = {};
public:
	auto const & getBuffer() const { return bm->getBuffer(); }
	auto const & getMemory() const { return bm->getMemory(); }
	void * getMapped() { return mapped; }
	
	VulkanBufferMemoryAndMapped(
		std::unique_ptr<VulkanBufferAndMemory> && bm_,
		void * && mapped_
	) :	bm(std::move(bm_)),
		mapped(std::move(mapped_))
	{}

	VulkanBufferMemoryAndMapped(VulkanBufferMemoryAndMapped && o)
	: 	bm(std::move(o.bm)),
		mapped(std::move(o.mapped))
	{}

	VulkanBufferMemoryAndMapped & operator=(VulkanBufferMemoryAndMapped && o) {
		bm = std::move(o.bm);
		mapped = std::move(o.mapped);
		return *this;
	}
};

namespace VulkanDeviceMakeFromStagingBuffer {
	auto create(
		vk::raii::PhysicalDevice const & physicalDevice,
		vk::raii::Device const & device,
		void const * const srcData,
		size_t bufferSize
	) {
		auto buffer = vk::raii::Buffer(
			device,
			vk::BufferCreateInfo()
				.setSize(bufferSize)
				.setUsage(vk::BufferUsageFlagBits::eTransferSrc)
				.setSharingMode(vk::SharingMode::eExclusive)
		);
		auto memRequirements = (*device).getBufferMemoryRequirements(*buffer);
		auto memory = vk::raii::DeviceMemory(
			device,
			vk::MemoryAllocateInfo()
				.setAllocationSize(memRequirements.size)
				.setMemoryTypeIndex(VulkanPhysicalDevice::findMemoryType(
					physicalDevice,
					memRequirements.memoryTypeBits,
					vk::MemoryPropertyFlagBits::eHostVisible
					| vk::MemoryPropertyFlagBits::eHostCoherent
				))
		);
		(*device).bindBufferMemory(
			*buffer,
			*memory,
			0
		);

		void * dstData = memory.mapMemory(
			0,
			bufferSize,
			vk::MemoryMapFlags{}
		);
		memcpy(dstData, srcData, (size_t)bufferSize);
		memory.unmapMemory();
		
		return std::make_unique<VulkanBufferAndMemory>(std::move(buffer), std::move(memory));
	}
};

namespace VulkanDeviceMemoryBuffer  {
	auto create(
		vk::raii::PhysicalDevice const & physicalDevice,
		vk::raii::Device const & device,
		vk::DeviceSize size,
		vk::BufferUsageFlags usage,
		vk::MemoryPropertyFlags properties
	) {
		auto buffer = vk::raii::Buffer(
			device,
			vk::BufferCreateInfo()
				.setFlags(vk::BufferCreateFlags())
				.setSize(size)
				.setUsage(usage)
				.setSharingMode(vk::SharingMode::eExclusive)
		);
		auto memRequirements = (*device).getBufferMemoryRequirements(*buffer);
		auto memory = vk::raii::DeviceMemory(
			device,
			vk::MemoryAllocateInfo()
				.setAllocationSize(memRequirements.size)
				.setMemoryTypeIndex(VulkanPhysicalDevice::findMemoryType(
					physicalDevice,
					memRequirements.memoryTypeBits,
					properties
				))
		);
		(*device).bindBufferMemory(
			*buffer,
			*memory,
			0
		);
		return std::make_unique<VulkanBufferAndMemory>(std::move(buffer), std::move(memory));
	}

	auto makeBufferFromStaged(
		vk::raii::PhysicalDevice const & physicalDevice,
		vk::raii::Device const & device,
		VulkanCommandPool const & commandPool,
		void const * const srcData,
		size_t bufferSize
	) {
		auto stagingBufferAndMemory = VulkanDeviceMakeFromStagingBuffer::create(
			physicalDevice,
			device,
			srcData,
			bufferSize
		);

		auto bufferAndMemory = VulkanDeviceMemoryBuffer::create(
			physicalDevice,
			device,
			bufferSize,
			vk::BufferUsageFlagBits::eTransferDst
			| vk::BufferUsageFlagBits::eVertexBuffer,
			vk::MemoryPropertyFlagBits::eDeviceLocal
		);
		
		commandPool.copyBuffer(
			*stagingBufferAndMemory->getBuffer(),
			*bufferAndMemory->getBuffer(),
			bufferSize
		);

		return bufferAndMemory;
	}
};

namespace VulkanDeviceMemoryImage {
	auto createImage(
		vk::raii::PhysicalDevice const & physicalDevice,
		vk::raii::Device const & device,
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
		auto image = vk::raii::Image(
			device,
			vk::ImageCreateInfo()
				.setImageType(vk::ImageType::e2D)
				.setFormat(format)
				.setExtent(vk::Extent3D(width, height, 1))
				.setMipLevels(mipLevels)
				.setArrayLayers(1)
				.setSamples(numSamples)
				.setTiling(tiling)
				.setUsage(usage)
				.setSharingMode(vk::SharingMode::eExclusive)
				.setInitialLayout(vk::ImageLayout::eUndefined)
		);

		auto memRequirements = image.getMemoryRequirements();
		auto imageMemory = vk::raii::DeviceMemory(
			device,
			vk::MemoryAllocateInfo()
				.setAllocationSize(memRequirements.size)
				.setMemoryTypeIndex(VulkanPhysicalDevice::findMemoryType(
					physicalDevice,
					memRequirements.memoryTypeBits,
					properties
				))
		);
		(*device).bindImageMemory(*image, *imageMemory, 0);
		return std::make_unique<VulkanImageAndMemory>(std::move(image), std::move(imageMemory));
	}

	auto makeTextureFromStaged(
		vk::raii::PhysicalDevice const & physicalDevice,
		vk::raii::Device const & device,
		VulkanCommandPool const & commandPool,
		void const * const srcData,
		size_t bufferSize,
		int texWidth,
		int texHeight,
		uint32_t mipLevels
	) {
		auto stagingBufferAndMemory = VulkanDeviceMakeFromStagingBuffer::create(
			physicalDevice,
			device,
			srcData,
			bufferSize
		);

		auto imageAndMemory = createImage(
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
			*imageAndMemory->getImage(),
			vk::ImageLayout::eUndefined,
			vk::ImageLayout::eTransferDstOptimal,
			mipLevels
		);
		commandPool.copyBufferToImage(
			*stagingBufferAndMemory->getBuffer(),
			*imageAndMemory->getImage(),
			(uint32_t)texWidth,
			(uint32_t)texHeight
		);
		/*
		commandPool.transitionImageLayout(
			*imageAndMemory->getImage(),
			vk::ImageLayout::eTransferDstOptimal,
			vk::ImageLayout::eShaderReadOnlyOptimal,
			mipLevels
		);
		*/
	
		//compiler tells me to remove this ... but won't that still call the dtor?
		//return std::move(imageAndMemory);
		return imageAndMemory;
	}
};

struct VulkanSwapChain {
protected:
	//owned
	vk::raii::SwapchainKHR obj;
	vk::raii::RenderPass renderPass;

	std::unique_ptr<VulkanImageAndMemory> depthImageAndMemory;
	std::unique_ptr<vk::raii::ImageView> depthImageView;
	
	std::unique_ptr<VulkanImageAndMemory> colorImageAndMemory;
	std::unique_ptr<vk::raii::ImageView> colorImageView;
public:
	vk::Extent2D extent;
	
	// I would combine these into one struct so they can be dtored together
	// but it seems vulkan wants vk::Images linear for its getter?
	std::vector<vk::Image> images;
	std::vector<std::unique_ptr<vk::raii::ImageView>> imageViews;
	std::vector<vk::raii::Framebuffer> framebuffers;
	
public:
	auto const & operator()() const { return obj; }
	auto const & getRenderPass() const { return renderPass; }

	// ************** from here on down, app-specific **************
	// but so are all the member variables so ...

	VulkanSwapChain(VulkanSwapChain && o) 
	: 	obj(std::move(o.obj)),
		renderPass(std::move(o.renderPass)),
		depthImageAndMemory(std::move(o.depthImageAndMemory)),
		depthImageView(std::move(o.depthImageView)),
		colorImageAndMemory(std::move(o.colorImageAndMemory)),
		colorImageView(std::move(o.colorImageView)),
		extent(std::move(o.extent)),
		images(std::move(o.images)),
		imageViews(std::move(o.imageViews)),
		framebuffers(std::move(o.framebuffers))
	{
	}

	VulkanSwapChain & operator=(VulkanSwapChain && o) {
	 	obj = std::move(o.obj);
		renderPass = std::move(o.renderPass);
		depthImageAndMemory = std::move(o.depthImageAndMemory);
		depthImageView = std::move(o.depthImageView);
		colorImageAndMemory = std::move(o.colorImageAndMemory);
		colorImageView = std::move(o.colorImageView);
		extent = std::move(o.extent);
		images = std::move(o.images);
		imageViews = std::move(o.imageViews);
		framebuffers = std::move(o.framebuffers);
		return *this;
	}

	//should I make this protected friend?
	VulkanSwapChain(
		vk::raii::SwapchainKHR && obj_,
		vk::raii::RenderPass && renderPass_,
		std::unique_ptr<VulkanImageAndMemory> && depthImageAndMemory_,
		std::unique_ptr<vk::raii::ImageView> && depthImageView_,
		std::unique_ptr<VulkanImageAndMemory> && colorImageAndMemory_,
		std::unique_ptr<vk::raii::ImageView> && colorImageView_,
		vk::Extent2D && extent_,
		std::vector<vk::Image> && images_,
		std::vector<std::unique_ptr<vk::raii::ImageView>> && imageViews_,
		std::vector<vk::raii::Framebuffer> && framebuffers_
	) : obj(std::move(obj_)),
		renderPass(std::move(renderPass_)),
		depthImageAndMemory(std::move(depthImageAndMemory_)),
		depthImageView(std::move(depthImageView_)),
		colorImageAndMemory(std::move(colorImageAndMemory_)),
		colorImageView(std::move(colorImageView_)),
		extent(std::move(extent_)),
		images(std::move(images_)),
		imageViews(std::move(imageViews_)),
		framebuffers(std::move(framebuffers_))
	{}

	//goes in a separate method because it puts a few objects on the stack while it builds vk objs
	static VulkanSwapChain create(
		Tensor::int2 screenSize,
		vk::raii::PhysicalDevice const & physicalDevice,
		vk::raii::Device const & device,
		vk::raii::SurfaceKHR const & surface,
		vk::SampleCountFlagBits msaaSamples
	) {
		auto swapChainSupport = VulkanPhysicalDevice::querySwapChainSupport(physicalDevice, surface);
		auto extent = chooseSwapExtent(screenSize, swapChainSupport.capabilities);

		// how come imageCount is one less than vkGetSwapchainImagesKHR gives?
		// maxImageCount == 0 means no max?
		uint32_t imageCount = swapChainSupport.capabilities.minImageCount + 1;
		if (swapChainSupport.capabilities.maxImageCount > 0) {
			imageCount = std::min(imageCount, swapChainSupport.capabilities.maxImageCount);
		}

		auto surfaceFormat = chooseSwapSurfaceFormat(swapChainSupport.formats);
		auto presentMode = chooseSwapPresentMode(swapChainSupport.presentModes);
		auto createInfo = vk::SwapchainCreateInfoKHR()
			.setSurface(*surface)
			.setMinImageCount(imageCount)
			.setImageFormat(surfaceFormat.format)
			.setImageColorSpace(surfaceFormat.colorSpace)
			.setImageExtent(extent)
			.setImageArrayLayers(1)
			.setImageUsage(vk::ImageUsageFlagBits::eColorAttachment)
			.setPreTransform(swapChainSupport.capabilities.currentTransform)
			.setCompositeAlpha(vk::CompositeAlphaFlagBitsKHR::eOpaque)
			.setPresentMode(presentMode)
			.setClipped(true);
		auto indices = VulkanPhysicalDevice::findQueueFamilies(physicalDevice, surface);
		auto queueFamilyIndices = Common::make_array<uint32_t>(
			(uint32_t)indices.graphicsFamily.value(),
			(uint32_t)indices.presentFamily.value()
		);
		if (indices.graphicsFamily != indices.presentFamily) {
			createInfo.setImageSharingMode(vk::SharingMode::eConcurrent);
			createInfo.setQueueFamilyIndices(queueFamilyIndices);
		} else {
			createInfo.setImageSharingMode(vk::SharingMode::eExclusive);
		}
		auto obj = vk::raii::SwapchainKHR(
			device,
			createInfo
		);

		std::vector<vk::Image> images;
		for (auto const & vkimage : obj.getImages()) {
			images.push_back(vk::Image(vkimage));
		}
		
		std::vector<std::unique_ptr<vk::raii::ImageView>> imageViews;
		for (size_t i = 0; i < images.size(); i++) {
			imageViews.push_back(createImageView(
				device,
				images[i],
				surfaceFormat.format,
				vk::ImageAspectFlagBits::eColor,
				1
			));
		}
	
		auto renderPass = VulkanRenderPass::create(
			physicalDevice,
			device,
			surfaceFormat.format,
			msaaSamples
		);
		
		//createColorResources
		auto colorFormat = surfaceFormat.format;
		
		auto colorImageAndMemory = VulkanDeviceMemoryImage::createImage(
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
		auto colorImageView = createImageView(
			device,
			*colorImageAndMemory->getImage(),
			colorFormat,
			vk::ImageAspectFlagBits::eColor,
			1
		);
		
		//createDepthResources
		auto depthFormat = VulkanPhysicalDevice::findDepthFormat(physicalDevice);
		
		auto depthImageAndMemory = VulkanDeviceMemoryImage::createImage(
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
		auto depthImageView = createImageView(
			device,
			*depthImageAndMemory->getImage(),
			depthFormat,
			vk::ImageAspectFlagBits::eDepth,
			1
		);
		
		//createFramebuffers
		std::vector<vk::raii::Framebuffer> framebuffers;
		for (size_t i = 0; i < imageViews.size(); i++) {
			auto attachments = Common::make_array(
				**colorImageView,
				**depthImageView,
				**imageViews[i]
			);
			framebuffers.push_back(
				vk::raii::Framebuffer(
					device,
					vk::FramebufferCreateInfo()
						.setRenderPass(*renderPass)
						.setAttachments(attachments)
						.setWidth(extent.width)
						.setHeight(extent.height)
						.setLayers(1)
				)
			);
		}
	
		return VulkanSwapChain(
			std::move(obj),
			std::move(renderPass),
			std::move(depthImageAndMemory),
			std::move(depthImageView),
			std::move(colorImageAndMemory),
			std::move(colorImageView),
			std::move(extent),
			std::move(images),
			std::move(imageViews),
			std::move(framebuffers)
		);
	}

public:
	static
	std::unique_ptr<vk::raii::ImageView>
	createImageView(
		vk::raii::Device const & device,
		vk::Image image,
		vk::Format format,
		vk::ImageAspectFlags aspectFlags,
		uint32_t mipLevels
	) {
		return std::make_unique<vk::raii::ImageView>(
			device,
			vk::ImageViewCreateInfo()
				.setImage(image)
				.setViewType(vk::ImageViewType::e2D)
				.setFormat(format)
				.setSubresourceRange(
					vk::ImageSubresourceRange()
						.setAspectMask(aspectFlags)
						.setLevelCount(mipLevels)
						.setLayerCount(1)
				)
		);
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

//only used by VulkanGraphicsPipeline's ctor
namespace VulkanShaderModule {
	auto fromFile(
		vk::raii::Device const & device,
		std::string const & filename
	) {
		auto code = Common::File::read(filename);
		return vk::raii::ShaderModule(
			device,
			vk::ShaderModuleCreateInfo()
				// why isn't there one method that does these two things?
				.setPCode((uint32_t const *)code.data())
				.setCodeSize(code.size())
		);
	}
};

struct VulkanGraphicsPipeline  {
protected:
	//owned:
	vk::raii::Pipeline obj;
	vk::raii::PipelineLayout pipelineLayout;
	vk::raii::DescriptorSetLayout descriptorSetLayout;
public:
	auto const & operator()() const { return obj; }
	auto const & getPipelineLayout() const { return pipelineLayout; }
	auto const & getDescriptorSetLayout() const { return descriptorSetLayout; }
	
	//TODO protect? 
	VulkanGraphicsPipeline(
		vk::raii::Pipeline && obj_,
		vk::raii::PipelineLayout && pipelineLayout_,
		vk::raii::DescriptorSetLayout && descriptorSetLayout_
	) : obj(std::move(obj_)),
		pipelineLayout(std::move(pipelineLayout_)),
		descriptorSetLayout(std::move(descriptorSetLayout_))
	{}

	//goes in a separate method because it puts a few objects on the stack while it builds vk objs
	static VulkanGraphicsPipeline create(
		vk::raii::PhysicalDevice const & physicalDevice,
		vk::raii::Device const & device,
		vk::raii::RenderPass const & renderPass,
		vk::SampleCountFlagBits msaaSamples
	) {
		// descriptorSetLayout is only used by graphicsPipeline
		auto bindings = Common::make_array(
			vk::DescriptorSetLayoutBinding()	//uboLayoutBinding
				.setBinding(0)
				.setDescriptorType(vk::DescriptorType::eUniformBuffer)
				.setDescriptorCount(1)
				.setStageFlags(vk::ShaderStageFlagBits::eVertex)
			,
			vk::DescriptorSetLayoutBinding()	//samplerLayoutBinding
				.setBinding(1)
				.setDescriptorType(vk::DescriptorType::eCombinedImageSampler)
				.setDescriptorCount(1)
				.setStageFlags(vk::ShaderStageFlagBits::eFragment)
			
		);
		auto descriptorSetLayout = vk::raii::DescriptorSetLayout(
			device,
			vk::DescriptorSetLayoutCreateInfo()
				.setBindings(bindings)
		);

		auto vertShaderModule = VulkanShaderModule::fromFile(device, "shader-vert.spv");
		auto fragShaderModule = VulkanShaderModule::fromFile(device, "shader-frag.spv");
		
		auto bindingDescriptions = Common::make_array(
			Vertex::getBindingDescription()
		);
		auto attributeDescriptions = Vertex::getAttributeDescriptions();
		auto vertexInputInfo = vk::PipelineVertexInputStateCreateInfo()
			.setVertexBindingDescriptions(bindingDescriptions)
			.setVertexAttributeDescriptions(attributeDescriptions);

		auto inputAssembly = vk::PipelineInputAssemblyStateCreateInfo()
			.setTopology(vk::PrimitiveTopology::eTriangleList)
			.setPrimitiveRestartEnable(false);

		auto viewportState = vk::PipelineViewportStateCreateInfo()
			.setViewportCount(1)
			.setScissorCount(1);

		auto rasterizer = vk::PipelineRasterizationStateCreateInfo()
			.setDepthClampEnable(false)
			.setRasterizerDiscardEnable(false)
			.setPolygonMode(vk::PolygonMode::eFill)
			//.cullMode = vk::CullModeFlagBits::eBack,
			//.frontFace = vk::FrontFace::eClockwise,
			//.frontFace = vk::FrontFace::eCounterClockwise,
			.setDepthBiasEnable(false)
			.setLineWidth(1);

		auto multisampling = vk::PipelineMultisampleStateCreateInfo()
			.setRasterizationSamples(msaaSamples)
			.setSampleShadingEnable(false);

		auto depthStencil = vk::PipelineDepthStencilStateCreateInfo()
			.setDepthTestEnable(true)
			.setDepthWriteEnable(true)
			.setDepthCompareOp(vk::CompareOp::eLess)
			.setDepthBoundsTestEnable(false)
			.setStencilTestEnable(false);

		auto colorBlendAttachment = vk::PipelineColorBlendAttachmentState()
			.setBlendEnable(false)
			.setColorWriteMask(
				vk::ColorComponentFlagBits::eR
				| vk::ColorComponentFlagBits::eG
				| vk::ColorComponentFlagBits::eB
				| vk::ColorComponentFlagBits::eA
			);

		auto colorBlending = vk::PipelineColorBlendStateCreateInfo()
			.setLogicOpEnable(false)
			.setLogicOp(vk::LogicOp::eCopy)
			.setAttachmentCount(1)
			.setPAttachments(&colorBlendAttachment)
			.setBlendConstants({0.f, 0.f, 0.f, 0.f});

		auto dynamicStates = Common::make_array(
			vk::DynamicState::eViewport,
			vk::DynamicState::eScissor
		);
		auto dynamicState = vk::PipelineDynamicStateCreateInfo()
			.setDynamicStates(dynamicStates);
		
		auto descriptorSetLayouts = Common::make_array(
			*descriptorSetLayout
		);
		auto pipelineLayout = vk::raii::PipelineLayout(
			device,
			vk::PipelineLayoutCreateInfo()
				.setSetLayouts(descriptorSetLayouts)
		);

		auto shaderStages = Common::make_array(
			vk::PipelineShaderStageCreateInfo()
				.setStage(vk::ShaderStageFlagBits::eVertex)
				.setModule(*vertShaderModule)
				.setPName("main"),	//"vert"	//GLSL uses 'main', but clspv doesn't allow 'main', so ...
			vk::PipelineShaderStageCreateInfo()
				.setStage(vk::ShaderStageFlagBits::eFragment)
				.setModule(*fragShaderModule)
				.setPName("main")	//"frag"
		);
		auto obj = vk::raii::Pipeline(
			device,
			nullptr,
			vk::GraphicsPipelineCreateInfo()
				.setStages(shaderStages)
				.setPVertexInputState(&vertexInputInfo)	//why it need to be a pointer?
				.setPInputAssemblyState(&inputAssembly)
				.setPViewportState(&viewportState)
				.setPRasterizationState(&rasterizer)
				.setPMultisampleState(&multisampling)
				.setPDepthStencilState(&depthStencil)
				.setPColorBlendState(&colorBlending)
				.setPDynamicState(&dynamicState)
				.setLayout(*pipelineLayout)
				.setRenderPass(*renderPass)
				.setSubpass(0)
				.setBasePipelineHandle(vk::Pipeline())
		);

		return VulkanGraphicsPipeline(
			std::move(obj),
			std::move(pipelineLayout),
			std::move(descriptorSetLayout)
		);
	}
};

struct VulkanMesh {
protected:
	std::unique_ptr<VulkanBufferAndMemory> vertexBufferAndMemory;
	std::unique_ptr<VulkanBufferAndMemory> indexBufferAndMemory;
	uint32_t numIndices = {};
public:
	auto const & getVertexes() const { return *vertexBufferAndMemory; }
	auto const & getIndexes() const { return *indexBufferAndMemory; }
	uint32_t getNumIndexes() const { return numIndices; }

	VulkanMesh(
		std::unique_ptr<VulkanBufferAndMemory> && vertexBufferAndMemory_,
		std::unique_ptr<VulkanBufferAndMemory> && indexBufferAndMemory_,
		uint32_t numIndices_
	) : vertexBufferAndMemory(std::move(vertexBufferAndMemory_)),
		indexBufferAndMemory(std::move(indexBufferAndMemory_)),
		numIndices(numIndices_)
	{}

	static VulkanMesh create(
		std::string const & modelPath,
		vk::raii::PhysicalDevice const & physicalDevice,
		vk::raii::Device const & device,
		VulkanCommandPool const & commandPool
	) {
		
		tinyobj::attrib_t attrib;
		std::vector<tinyobj::shape_t> shapes;
		std::vector<tinyobj::material_t> materials;
		std::string warn, err;
		if (!tinyobj::LoadObj(&attrib, &shapes, &materials, &warn, &err, modelPath.c_str())) {
			throw Common::Exception() << warn << err;
		}

		std::vector<Vertex> vertices;
		std::vector<uint32_t> indices;
		std::unordered_map<Vertex, uint32_t> uniqueVertices;
		for (auto const & shape : shapes) {
			for (auto const & index : shape.mesh.indices) {
				Vertex vertex;
				vertex.pos = {
					attrib.vertices[3 * index.vertex_index + 0],
					attrib.vertices[3 * index.vertex_index + 1],
					attrib.vertices[3 * index.vertex_index + 2]
				};

				vertex.color = {1.0f, 1.0f, 1.0f};
				
				vertex.texCoord = {
					attrib.texcoords[2 * index.texcoord_index + 0],
					1.0f - attrib.texcoords[2 * index.texcoord_index + 1]
				};


				if (uniqueVertices.count(vertex) == 0) {
					uniqueVertices[vertex] = (uint32_t)vertices.size();
					vertices.push_back(vertex);
				}

				indices.push_back(uniqueVertices[vertex]);
			}
		}
		
		auto vertexBufferAndMemory = VulkanDeviceMemoryBuffer::makeBufferFromStaged(
			physicalDevice,
			device,
			commandPool,
			vertices.data(),
			sizeof(vertices[0]) * vertices.size()
		);
		uint32_t numIndices = indices.size();
		auto indexBufferAndMemory = VulkanDeviceMemoryBuffer::makeBufferFromStaged(
			physicalDevice,
			device,
			commandPool,
			indices.data(),
			sizeof(indices[0]) * indices.size()
		);
		return VulkanMesh(
			std::move(vertexBufferAndMemory),
			std::move(indexBufferAndMemory),
			std::move(numIndices)
		);
	}
};

// so I don't have to prefix all my fields and names
struct VulkanCommon {
protected:
	static constexpr std::string modelPath = "viking_room.obj";
	static constexpr std::string texturePath = "viking_room.png";
	static constexpr int maxFramesInFlight = 2;

	::SDLApp::SDLApp const * app = {};	// points back to the owner

	//this is used by instance, so must go above instance
#if 0	// extension not found on my vulkan implementation
	static constexpr bool const enableValidationLayers = true;
#else
	static constexpr bool const enableValidationLayers = false;
#endif

	//this is used by physicalDevice, so has to go above physicalDevice
protected:
	std::vector<char const *> const deviceExtensions = {
		VK_KHR_SWAPCHAIN_EXTENSION_NAME
	};


	vk::raii::Context ctx;				// hmm, in raii but not in non-raii oop c++ hpp?
	vk::raii::Instance instance;
	std::unique_ptr<vk::raii::DebugUtilsMessengerEXT> debug;	// optional
	vk::raii::SurfaceKHR surface;
	vk::raii::PhysicalDevice physicalDevice;
	vk::SampleCountFlagBits msaaSamples = vk::SampleCountFlagBits::e1;
	VulkanDevice device;
	VulkanSwapChain swapChain;
	VulkanGraphicsPipeline graphicsPipeline;
	VulkanCommandPool commandPool;
	
	//these two are initialized together:
	uint32_t mipLevels = {};
	std::unique_ptr<VulkanImageAndMemory> textureImageAndMemory;
	
	std::unique_ptr<vk::raii::ImageView> textureImageView;
	vk::raii::Sampler textureSampler;

	VulkanMesh mesh;
	std::vector<VulkanBufferMemoryAndMapped> uniformBuffers;
	vk::raii::DescriptorPool descriptorPool;
	
	// each of these, there are one per number of frames in flight
	std::vector<vk::raii::DescriptorSet> descriptorSets;
	std::vector<vk::raii::CommandBuffer> commandBuffers;
	std::vector<vk::raii::Semaphore> imageAvailableSemaphores;
	std::vector<vk::raii::Semaphore> renderFinishedSemaphores;
	std::vector<vk::raii::Fence> inFlightFences;
	
	uint32_t currentFrame = {};
	
	bool framebufferResized = {};
public:
	void setFramebufferResized() { framebufferResized = true; }

protected:	
	// has to be called only after all fields above 'swapChain' have been initialized
	VulkanSwapChain createSwapChain() {
		return VulkanSwapChain::create(
			app->getScreenSize(),
			physicalDevice,
			device(),
			surface,
			msaaSamples
		);
	}

public:
	VulkanCommon(
		::SDLApp::SDLApp const * const app_
	) : app(app_),
		instance([this](){
			if (enableValidationLayers && !checkValidationLayerSupport()) {
				throw Common::Exception() << "validation layers requested, but not available!";
			}
			
			// hmm, maybe instance should be a shared_ptr and then passed to debug, surface, and physicalDevice ?
			return VulkanInstance::create(ctx, app, enableValidationLayers);
		}()),
		debug(!enableValidationLayers 
			? std::unique_ptr<vk::raii::DebugUtilsMessengerEXT>{} 
			: std::make_unique<vk::raii::DebugUtilsMessengerEXT>(
				instance,
				vk::DebugUtilsMessengerCreateInfoEXT()
					.setMessageSeverity(
						vk::DebugUtilsMessageSeverityFlagBitsEXT::eVerbose 
							| vk::DebugUtilsMessageSeverityFlagBitsEXT::eWarning
							| vk::DebugUtilsMessageSeverityFlagBitsEXT::eError
					)
					.setMessageType(
						vk::DebugUtilsMessageTypeFlagBitsEXT::eGeneral
							| vk::DebugUtilsMessageTypeFlagBitsEXT::eValidation
							| vk::DebugUtilsMessageTypeFlagBitsEXT::ePerformance
					)
					.setPfnUserCallback(
						[](
							VkDebugUtilsMessageSeverityFlagBitsEXT messageSeverity,
							VkDebugUtilsMessageTypeFlagsEXT messageType,
							VkDebugUtilsMessengerCallbackDataEXT const * pCallbackData,
							void * pUserData
						) -> VKAPI_ATTR VkBool32 VKAPI_CALL
						{
							std::cerr << "validation layer: " << pCallbackData->pMessage << std::endl;
							return VK_FALSE;
						}
					)
			)
		),
		surface([this](){
			VkSurfaceKHR h = {};
			SDL_VULKAN_SAFE(SDL_Vulkan_CreateSurface, app->getWindow(), *instance, &h);
			return vk::raii::SurfaceKHR(instance, h);
		}()),
		physicalDevice(VulkanPhysicalDevice::create(
			instance,
			surface,
			deviceExtensions
		)),
		msaaSamples(VulkanPhysicalDevice::getMaxUsableSampleCount(physicalDevice)),
		device(
			physicalDevice,
			deviceExtensions,
			enableValidationLayers,
			VulkanPhysicalDevice::findQueueFamilies(physicalDevice, surface)
		),
		swapChain(createSwapChain()),
		graphicsPipeline(VulkanGraphicsPipeline::create(
			physicalDevice,
			device(),
			swapChain.getRenderPass(),
			msaaSamples
		)),
		commandPool([this](){
			auto queueFamilyIndices = VulkanPhysicalDevice::findQueueFamilies(
				physicalDevice,
				surface
			);
			return VulkanCommandPool(
				device(),
				device.getGraphicsQueue(),
				vk::CommandPoolCreateInfo()
					.setFlags(vk::CommandPoolCreateFlagBits::eResetCommandBuffer)
					.setQueueFamilyIndex(queueFamilyIndices.graphicsFamily.value())
			);	
		}()),
		textureImageAndMemory(createTextureImage()),
		textureImageView(VulkanSwapChain::createImageView(
			device(),
			*textureImageAndMemory->getImage(),
			vk::Format::eR8G8B8A8Srgb,
			vk::ImageAspectFlagBits::eColor,
			mipLevels
		)),
		textureSampler(
			device(),
			vk::SamplerCreateInfo()
				.setMagFilter(vk::Filter::eLinear)
				.setMinFilter(vk::Filter::eLinear)
				.setMipmapMode(vk::SamplerMipmapMode::eLinear)
				.setAddressModeU(vk::SamplerAddressMode::eRepeat)
				.setAddressModeV(vk::SamplerAddressMode::eRepeat)
				.setAddressModeW(vk::SamplerAddressMode::eRepeat)
				.setAnisotropyEnable(true)
				.setMaxAnisotropy(physicalDevice.getProperties().limits.maxSamplerAnisotropy)
				.setCompareEnable(false)
				.setCompareOp(vk::CompareOp::eAlways)
				.setMinLod(0)
				.setMaxLod((float)mipLevels)
				.setBorderColor(vk::BorderColor::eIntOpaqueBlack)
				.setUnnormalizedCoordinates(false)
		),
		mesh(VulkanMesh::create(
			modelPath,
			physicalDevice,
			device(),
			commandPool
		)),
		uniformBuffers([this](){
			// https://stackoverflow.com/questions/51307168/how-to-fill-a-c-container-using-a-lambda-function
			std::vector<VulkanBufferMemoryAndMapped> tmp;
			std::generate_n(
				std::back_inserter(tmp),
				maxFramesInFlight,
				[this](){
					auto b = VulkanDeviceMemoryBuffer::create(
						physicalDevice,
						device(),
						sizeof(UniformBufferObject),
						vk::BufferUsageFlagBits::eUniformBuffer,
						vk::MemoryPropertyFlagBits::eHostVisible
						| vk::MemoryPropertyFlagBits::eHostCoherent
					);
					auto m = b->getMemory().mapMemory(
						0,
						sizeof(UniformBufferObject),
						vk::MemoryMapFlags{}
					);
					return VulkanBufferMemoryAndMapped(std::move(b), std::move(m));		
				}
			);
			return tmp;	
		}()),
		descriptorPool([this](){
			auto poolSizes = Common::make_array(
				vk::DescriptorPoolSize()
					.setType(vk::DescriptorType::eUniformBuffer)
					.setDescriptorCount(maxFramesInFlight),
				vk::DescriptorPoolSize()
					.setType(vk::DescriptorType::eCombinedImageSampler)
					.setDescriptorCount(maxFramesInFlight)
			);
			return vk::raii::DescriptorPool(
				device(),
				vk::DescriptorPoolCreateInfo()
					.setMaxSets(maxFramesInFlight)
					//why aren't these two merged into one function?
					.setPoolSizeCount(poolSizes.size())
					.setPPoolSizes(poolSizes.data())
			);
		}()),
		descriptorSets(createDescriptorSets()),
		commandBuffers(
			// TODO this matches 'VulkanSingleTimeCommand' ctor
			device().allocateCommandBuffers(
				vk::CommandBufferAllocateInfo()
					.setCommandPool(*commandPool())
					.setLevel(vk::CommandBufferLevel::ePrimary)
					.setCommandBufferCount(maxFramesInFlight)
			)
			// end part that matches
		)
	{
		initSyncObjects();
	}

public:
	~VulkanCommon() {
		// vector of unique pointers, can't use `= {}`, gotta use `.clear()`
		commandBuffers.clear();
		imageAvailableSemaphores.clear();
		renderFinishedSemaphores.clear();
		inFlightFences.clear();
	}

protected:
	// this is out of place
	static bool checkValidationLayerSupport() {
		auto availableLayers = vk::enumerateInstanceLayerProperties();
		for (char const * const layerName : validationLayers) {
			bool layerFound = false;
			for (auto const & layerProperties : availableLayers) {
				// hmm, why does vulkan hpp use array<char> instead of string?
				if (!strcmp(layerName, layerProperties.layerName.data())) {
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


protected:
	//also initializes mipLevels
	decltype(textureImageAndMemory)
	createTextureImage() {
		auto image = std::dynamic_pointer_cast<Image::Image>(Image::system->read(texturePath));
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
	
		auto textureImageAndMemory = VulkanDeviceMemoryImage::makeTextureFromStaged(
			physicalDevice,
			device(),
			commandPool,
			srcData,
			bufferSize,
			texSize.x,
			texSize.y,
			mipLevels
		);
	
		generateMipmaps(
			textureImageAndMemory->getImage(),
			vk::Format::eR8G8B8A8Srgb,
			texSize.x,
			texSize.y,
			mipLevels
		);

		return textureImageAndMemory;
	}

	void generateMipmaps(
		vk::raii::Image const & image,
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
			device(),
			device.getGraphicsQueue(),
			commandPool()
		);

		auto barrier = vk::ImageMemoryBarrier()
			.setSrcQueueFamilyIndex(VK_QUEUE_FAMILY_IGNORED)
			.setDstQueueFamilyIndex(VK_QUEUE_FAMILY_IGNORED)
			.setImage(*image)
			.setSubresourceRange(
				vk::ImageSubresourceRange()
					.setAspectMask(vk::ImageAspectFlagBits::eColor)
					.setLevelCount(1)
					.setLayerCount(1)
			);
		
		int32_t mipWidth = texWidth;
		int32_t mipHeight = texHeight;

		for (uint32_t i = 1; i < mipLevels; i++) {
			barrier.subresourceRange.setBaseMipLevel(i - 1);
			barrier
				.setOldLayout(vk::ImageLayout::eTransferDstOptimal)
				.setNewLayout(vk::ImageLayout::eTransferSrcOptimal)
				.setSrcAccessMask(vk::AccessFlagBits::eTransferWrite)
				.setDstAccessMask(vk::AccessFlagBits::eTransferRead);
			commandBuffer().pipelineBarrier(
				vk::PipelineStageFlagBits::eTransfer,
				vk::PipelineStageFlagBits::eTransfer,
				vk::DependencyFlags{},
				{},
				{},
				barrier
			);

			auto blit = vk::ImageBlit()
				.setSrcSubresource(
					vk::ImageSubresourceLayers()
						.setAspectMask(vk::ImageAspectFlagBits::eColor)
						.setMipLevel(i - 1)
						.setLayerCount(1)
				)
				.setSrcOffsets(Common::make_array(
					vk::Offset3D(),
					vk::Offset3D(mipWidth, mipHeight, 1)
				))
				.setDstSubresource(
					vk::ImageSubresourceLayers()
						.setAspectMask(vk::ImageAspectFlagBits::eColor)
						.setMipLevel(i)
						.setLayerCount(1)
				)
				.setDstOffsets(Common::make_array(
					vk::Offset3D(),
					vk::Offset3D(
						mipWidth > 1 ? mipWidth / 2 : 1,
						mipHeight > 1 ? mipHeight / 2 : 1,
						1
					)
				));
			
			commandBuffer().blitImage(
				*image,
				vk::ImageLayout::eTransferSrcOptimal,
				*image,
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
		device().waitIdle();
		swapChain = createSwapChain();
	}

	void recordCommandBuffer(
		vk::raii::CommandBuffer const & commandBuffer,
		uint32_t imageIndex
	) {
		// TODO this part matches VulkanSingleTimeCommand ctor
		commandBuffer.begin(vk::CommandBufferBeginInfo{});
		// end part that matches

		auto clearValues = Common::make_array(
			vk::ClearValue()
				.setColor(vk::ClearColorValue(Common::make_array<float>(0.f, 0.f, 0.f, 1.f))),
			vk::ClearValue()
				.setDepthStencil(vk::ClearDepthStencilValue(1.f, 0))
		);
		commandBuffer.beginRenderPass(
			vk::RenderPassBeginInfo()
				.setRenderPass(*swapChain.getRenderPass())
				.setFramebuffer(*swapChain.framebuffers[imageIndex])
				.setRenderArea(
					vk::Rect2D()
						.setExtent(swapChain.extent)
				)
				.setClearValues(clearValues),
			vk::SubpassContents::eInline
		);

		{
			commandBuffer.bindPipeline(
				vk::PipelineBindPoint::eGraphics,
				*graphicsPipeline()
			);

			commandBuffer.setViewport(
				0,
				Common::make_array(
					vk::Viewport()
						.setWidth(swapChain.extent.width)
						.setHeight(swapChain.extent.height)
						.setMinDepth(0)
						.setMaxDepth(1)
				)
			);

			commandBuffer.setScissor(
				0,
				Common::make_array(
					vk::Rect2D()
						.setExtent(swapChain.extent)
				)
			);

			commandBuffer.bindVertexBuffers(
				0,
				Common::make_array(
					*mesh.getVertexes().getBuffer()
				),
				Common::make_array<vk::DeviceSize>(0)
			);

			commandBuffer.bindIndexBuffer(
				*mesh.getIndexes().getBuffer(),
				0,
				vk::IndexType::eUint32
			);

			commandBuffer.bindDescriptorSets(
				vk::PipelineBindPoint::eGraphics,
				*graphicsPipeline.getPipelineLayout(),
				0,
				*descriptorSets[currentFrame],
				{}
			);

			commandBuffer.drawIndexed(
				mesh.getNumIndexes(),
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
		for (size_t i = 0; i < maxFramesInFlight; i++) {
			imageAvailableSemaphores.push_back(device().createSemaphore({}));
			renderFinishedSemaphores.push_back(device().createSemaphore({}));
			inFlightFences.push_back(device().createFence(
				vk::FenceCreateInfo()
					.setFlags(vk::FenceCreateFlagBits::eSignaled)
			));
		}
	}

protected:
	decltype(descriptorSets) createDescriptorSets() {
		std::vector<vk::DescriptorSetLayout> layouts(
			maxFramesInFlight,
			*graphicsPipeline.getDescriptorSetLayout()
		);
		auto descriptorSets = device().allocateDescriptorSets(
			vk::DescriptorSetAllocateInfo()
				.setDescriptorPool(*descriptorPool)
				.setDescriptorSetCount(maxFramesInFlight)
				.setSetLayouts(layouts)
		);

		for (size_t i = 0; i < maxFramesInFlight; i++) {
			auto bufferInfo = vk::DescriptorBufferInfo()
				.setBuffer(*uniformBuffers[i].getBuffer())
				.setRange(sizeof(UniformBufferObject));
			auto imageInfo = vk::DescriptorImageInfo()
				.setSampler(*textureSampler)
				.setImageView(**textureImageView)
				.setImageLayout(vk::ImageLayout::eShaderReadOnlyOptimal);
			auto descriptorWrites = Common::make_array(
				vk::WriteDescriptorSet()
					.setDstSet(*descriptorSets[i])
					.setDstBinding(0)
					.setDescriptorCount(1)
					.setDescriptorType(vk::DescriptorType::eUniformBuffer)
					.setBufferInfo(bufferInfo)
				,
				vk::WriteDescriptorSet()
					.setDstSet(*descriptorSets[i])
					.setDstBinding(1)
					.setDescriptorCount(1)
					.setDescriptorType(vk::DescriptorType::eCombinedImageSampler)
					.setImageInfo(imageInfo)
			);
			device().updateDescriptorSets(
				descriptorWrites,
				{}
			);
		}

		return descriptorSets;
	}

protected:
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
			(float)swapChain.extent.width / (float)swapChain.extent.height,
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
		memcpy(uniformBuffers[currentFrame_].getMapped(), &ubo, sizeof(ubo));
	}
public:
	void drawFrame() {
		{
			auto fences = Common::make_array(
				*inFlightFences[currentFrame]
			);
			auto result = device().waitForFences(
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
			vk::Result result;
			std::tie(result, imageIndex) = device().acquireNextImage2KHR(
				vk::AcquireNextImageInfoKHR()
					.setSwapchain(*swapChain())
					.setTimeout(UINT64_MAX)
					.setSemaphore(*imageAvailableSemaphores[currentFrame])
			);
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

		device().resetFences(
			Common::make_array(
				*inFlightFences[currentFrame]
			)
		);

		{
			auto const & o = commandBuffers[currentFrame];
			o.reset();
			recordCommandBuffer(o, imageIndex);
		}

		auto waitSemaphores = Common::make_array(
			*imageAvailableSemaphores[currentFrame]
		);
		auto waitStages = Common::make_array<vk::PipelineStageFlags>(
			vk::PipelineStageFlagBits::eColorAttachmentOutput
		);
		static_assert(waitSemaphores.size() == waitStages.size());
		
		auto signalSemaphores = Common::make_array(
			*renderFinishedSemaphores[currentFrame]
		);

		// static assert sizes match?
		auto cmdBufs = Common::make_array(
			*commandBuffers[currentFrame]
		);
		device.getGraphicsQueue().submit(
			vk::SubmitInfo()
			.setWaitSemaphores(waitSemaphores)
			.setWaitDstStageMask(waitStages)
			.setCommandBuffers(cmdBufs)
			.setSignalSemaphores(signalSemaphores),
			*inFlightFences[currentFrame]
		);
		
		auto swapChains = Common::make_array(
			*swapChain()
		);
		auto result = device.getPresentQueue().presentKHR(
			vk::PresentInfoKHR()
			.setWaitSemaphores(signalSemaphores)
			.setSwapchains(swapChains)
			.setPImageIndices(&imageIndex)
		);
		if (result == vk::Result::eErrorOutOfDateKHR || 
			result == vk::Result::eSuboptimalKHR || 
			framebufferResized
		) {
			framebufferResized = false;
			recreateSwapChain();
		} else if (result != vk::Result::eSuccess) {
			throw Common::Exception() << "vk::QueuePresentKHR failed: " << result;
		}

		currentFrame = (currentFrame + 1) % maxFramesInFlight;
	}
public:
	void loopDone() {
		device().waitIdle();
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
