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

namespace VulkanDevice {
	auto create(
		vk::raii::PhysicalDevice const & physicalDevice,
		vk::raii::SurfaceKHR const & surface,
		std::vector<char const *> const & deviceExtensions,
		bool enableValidationLayers
	) {
		auto indices = VulkanPhysicalDevice::findQueueFamilies(physicalDevice, surface);

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
		std::unique_ptr<vk::raii::Device> device = std::make_unique<vk::raii::Device>(
			physicalDevice,
			vk::DeviceCreateInfo()
				.setQueueCreateInfos(queueCreateInfos)
				.setPEnabledLayerNames(thisValidationLayers)
				.setPEnabledExtensionNames(deviceExtensions)
				.setPEnabledFeatures(&deviceFeatures)
		);
		std::unique_ptr<vk::raii::Queue> graphicsQueue = std::make_unique<vk::raii::Queue>(*device, indices.graphicsFamily.value(), 0);
		std::unique_ptr<vk::raii::Queue> presentQueue = std::make_unique<vk::raii::Queue>(*device, indices.presentFamily.value(), 0);
		return std::make_tuple(std::move(device), std::move(graphicsQueue), std::move(presentQueue));
	}
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
		return std::make_unique<vk::raii::RenderPass>(
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
	vk::raii::Device const & device;
	vk::raii::Queue const & queue;
	vk::raii::CommandPool const & commandPool;

public:
	auto const & operator()() const { return cmds[0]; }

	VulkanSingleTimeCommand(
		vk::raii::Device const & device_,
		vk::raii::Queue const & queue_,
		vk::raii::CommandPool const & commandPool_
	) : 
		cmds(
			device_.allocateCommandBuffers(
	//			commandPool_,
				vk::CommandBufferAllocateInfo()
					.setCommandPool(*commandPool_)
					.setLevel(vk::CommandBufferLevel::ePrimary)
					.setCommandBufferCount(1)
			)
		),
		device(device_),
		queue(queue_),
		commandPool(commandPool_)
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

struct VulkanCommandPool  {
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

namespace VulkanDeviceMakeFromStagingBuffer {
	std::pair<
		std::unique_ptr<vk::raii::Buffer>,
		std::unique_ptr<vk::raii::DeviceMemory>
	>
	create(
		vk::raii::PhysicalDevice const & physicalDevice,
		vk::raii::Device const & device,
		void const * const srcData,
		size_t bufferSize
	) {
		auto buffer = std::make_unique<vk::raii::Buffer>(
			device,
			vk::BufferCreateInfo()
				.setSize(bufferSize)
				.setUsage(vk::BufferUsageFlagBits::eTransferSrc)
				.setSharingMode(vk::SharingMode::eExclusive)
		);
		auto memRequirements = (*device).getBufferMemoryRequirements(**buffer);
		auto memory = std::make_unique<vk::raii::DeviceMemory>(
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
			**buffer,
			**memory,
			0
		);

		void * dstData = memory->mapMemory(
			0,
			bufferSize,
			vk::MemoryMapFlags{}
		);
		memcpy(dstData, srcData, (size_t)bufferSize);
		memory->unmapMemory();
		
		return std::make_pair(std::move(buffer), std::move(memory));
	}
};

namespace VulkanDeviceMemoryBuffer  {
	std::pair<
		std::unique_ptr<vk::raii::Buffer>,
		std::unique_ptr<vk::raii::DeviceMemory>
	>
	create(
		vk::raii::PhysicalDevice const & physicalDevice,
		vk::raii::Device const & device,
		vk::DeviceSize size,
		vk::BufferUsageFlags usage,
		vk::MemoryPropertyFlags properties
	) {
		auto buffer = std::make_unique<vk::raii::Buffer>(
			device,
			vk::BufferCreateInfo()
				.setFlags(vk::BufferCreateFlags())
				.setSize(size)
				.setUsage(usage)
				.setSharingMode(vk::SharingMode::eExclusive)
		);
		auto memRequirements = (*device).getBufferMemoryRequirements(**buffer);
		auto memory = std::make_unique<vk::raii::DeviceMemory>(
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
			**buffer,
			**memory,
			0
		);
		return std::make_pair(std::move(buffer), std::move(memory));
	}

	std::pair<
		std::unique_ptr<vk::raii::Buffer>,
		std::unique_ptr<vk::raii::DeviceMemory>
	>
	makeBufferFromStaged(
		vk::raii::PhysicalDevice const & physicalDevice,
		vk::raii::Device const & device,
		VulkanCommandPool const & commandPool,
		void const * const srcData,
		size_t bufferSize
	) {
		std::unique_ptr<vk::raii::Buffer> stagingBuffer;
		std::unique_ptr<vk::raii::DeviceMemory> stagingBufferMemory;
		std::tie(stagingBuffer, stagingBufferMemory) 
		= VulkanDeviceMakeFromStagingBuffer::create(
			physicalDevice,
			device,
			srcData,
			bufferSize
		);

		std::unique_ptr<vk::raii::Buffer> buffer;
		std::unique_ptr<vk::raii::DeviceMemory> memory;
		std::tie(buffer, memory) = VulkanDeviceMemoryBuffer::create(
			physicalDevice,
			device,
			bufferSize,
			vk::BufferUsageFlagBits::eTransferDst
			| vk::BufferUsageFlagBits::eVertexBuffer,
			vk::MemoryPropertyFlagBits::eDeviceLocal
		);
		
		commandPool.copyBuffer(
			**stagingBuffer,
			**buffer,
			bufferSize
		);

		return std::make_pair(std::move(buffer), std::move(memory));
	}
};

namespace VulkanDeviceMemoryImage {

	std::pair<
		std::unique_ptr<vk::raii::Image>,
		std::unique_ptr<vk::raii::DeviceMemory>
	>
	createImage(
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
		std::unique_ptr<vk::raii::Image> image = std::make_unique<vk::raii::Image>(
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

		auto memRequirements = image->getMemoryRequirements();
		std::unique_ptr<vk::raii::DeviceMemory> imageMemory = std::make_unique<vk::raii::DeviceMemory>(
			device,
			vk::MemoryAllocateInfo()
				.setAllocationSize(memRequirements.size)
				.setMemoryTypeIndex(VulkanPhysicalDevice::findMemoryType(
					physicalDevice,
					memRequirements.memoryTypeBits,
					properties
				))
		);
		(*device).bindImageMemory(**image, **imageMemory, 0);
		return std::make_pair(std::move(image), std::move(imageMemory));
	}

	std::pair<
		std::unique_ptr<vk::raii::Image>,
		std::unique_ptr<vk::raii::DeviceMemory>
	>
	makeTextureFromStaged(
		vk::raii::PhysicalDevice const & physicalDevice,
		vk::raii::Device const & device,
		VulkanCommandPool const & commandPool,
		void const * const srcData,
		size_t bufferSize,
		int texWidth,
		int texHeight,
		uint32_t mipLevels
	) {
		std::unique_ptr<vk::raii::Buffer> stagingBuffer;
		std::unique_ptr<vk::raii::DeviceMemory> stagingBufferMemory;
		std::tie(stagingBuffer, stagingBufferMemory) 
		= VulkanDeviceMakeFromStagingBuffer::create(
			physicalDevice,
			device,
			srcData,
			bufferSize
		);
		
		std::unique_ptr<vk::raii::Image> image;
		std::unique_ptr<vk::raii::DeviceMemory> imageMemory;
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
			**image,
			vk::ImageLayout::eUndefined,
			vk::ImageLayout::eTransferDstOptimal,
			mipLevels
		);
		commandPool.copyBufferToImage(
			**stagingBuffer,
			**image,
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
		
		return std::make_pair(std::move(image), std::move(imageMemory));
	}

};

struct VulkanSwapChain {
protected:
	std::unique_ptr<vk::raii::SwapchainKHR> obj;
	//owned
	std::unique_ptr<vk::raii::RenderPass> renderPass;

	std::unique_ptr<vk::raii::Image> depthImage;
	std::unique_ptr<vk::raii::DeviceMemory> depthImageMemory;
	std::unique_ptr<vk::raii::ImageView> depthImageView;
	
	std::unique_ptr<vk::raii::Image> colorImage;
	std::unique_ptr<vk::raii::DeviceMemory> colorImageMemory;
	std::unique_ptr<vk::raii::ImageView> colorImageView;
	
	// hold for this class lifespan
	vk::raii::Device const & device;
public:
	vk::Extent2D extent;
	
	// I would combine these into one struct so they can be dtored together
	// but it seems vulkan wants vk::Images linear for its getter?
	std::vector<vk::Image> images;
	std::vector<std::unique_ptr<vk::raii::ImageView>> imageViews;
	std::vector<vk::raii::Framebuffer> framebuffers;
	
public:
	auto const & operator()() const { return *obj; }
	auto const & getRenderPass() const { return *renderPass; }

	~VulkanSwapChain() {
		depthImageView = {};
		depthImageMemory = {};
		depthImage = {};
		
		colorImageView = {};
		colorImageMemory = {};
		colorImage = {};
		
		framebuffers.clear();
		imageViews.clear();
		obj = {};
	}

	// ************** from here on down, app-specific **************
	// but so are all the member variables so ...

	VulkanSwapChain(
		Tensor::int2 screenSize,
		vk::raii::PhysicalDevice const & physicalDevice,
		vk::raii::Device const & device_,
		vk::raii::SurfaceKHR const & surface,
		vk::SampleCountFlagBits msaaSamples
	) : device(device_) {
		auto swapChainSupport = VulkanPhysicalDevice::querySwapChainSupport(physicalDevice, surface);
		auto surfaceFormat = chooseSwapSurfaceFormat(swapChainSupport.formats);
		auto presentMode = chooseSwapPresentMode(swapChainSupport.presentModes);
		extent = chooseSwapExtent(screenSize, swapChainSupport.capabilities);

		// how come imageCount is one less than vkGetSwapchainImagesKHR gives?
		// maxImageCount == 0 means no max?
		uint32_t imageCount = swapChainSupport.capabilities.minImageCount + 1;
		if (swapChainSupport.capabilities.maxImageCount > 0) {
			imageCount = std::min(imageCount, swapChainSupport.capabilities.maxImageCount);
		}

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
		obj = std::make_unique<vk::raii::SwapchainKHR>(
			device,
			createInfo
		);

		for (auto const & vkimage : obj->getImages()) {
			images.push_back(vk::Image(vkimage));
		}
		for (size_t i = 0; i < images.size(); i++) {
			imageViews.push_back(createImageView(
				images[i],
				surfaceFormat.format,
				vk::ImageAspectFlagBits::eColor,
				1
			));
		}
	
		renderPass = VulkanRenderPass::create(
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
			**colorImage,
			colorFormat,
			vk::ImageAspectFlagBits::eColor,
			1
		);
		
		//createDepthResources
		auto depthFormat = VulkanPhysicalDevice::findDepthFormat(physicalDevice);
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
			**depthImage,
			depthFormat,
			vk::ImageAspectFlagBits::eDepth,
			1
		);
		
		//createFramebuffers
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
						.setRenderPass(**renderPass)
						.setAttachments(attachments)
						.setWidth(extent.width)
						.setHeight(extent.height)
						.setLayers(1)
				)
			);
		}
	}

public:
	std::unique_ptr<vk::raii::ImageView> createImageView(
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
	std::unique_ptr<vk::raii::Pipeline> obj;
	std::unique_ptr<vk::raii::PipelineLayout> pipelineLayout;
	std::unique_ptr<vk::raii::DescriptorSetLayout> descriptorSetLayout;
	
	//held:
	vk::raii::Device const & device;				//held for dtor
public:
	auto const & operator()() const { return obj; }
	auto const & getPipelineLayout() const { return pipelineLayout; }
	auto const & getDescriptorSetLayout() const { return descriptorSetLayout; }

	VulkanGraphicsPipeline(
		vk::raii::PhysicalDevice const & physicalDevice,
		vk::raii::Device const & device_,
		vk::raii::RenderPass const & renderPass,
		vk::SampleCountFlagBits msaaSamples
	) : device(device_) {
		
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
		descriptorSetLayout = std::make_unique<vk::raii::DescriptorSetLayout>(
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
			**descriptorSetLayout
		);
		pipelineLayout = std::make_unique<vk::raii::PipelineLayout>(
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
		obj = std::make_unique<vk::raii::Pipeline>(
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
				.setLayout(**pipelineLayout)
				.setRenderPass(*renderPass)
				.setSubpass(0)
				.setBasePipelineHandle(vk::Pipeline())
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

	std::unique_ptr<vk::raii::Device> device;
	std::unique_ptr<vk::raii::Queue> graphicsQueue;
	std::unique_ptr<vk::raii::Queue> presentQueue;
	std::unique_ptr<VulkanSwapChain> swapChain;
	std::unique_ptr<VulkanGraphicsPipeline> graphicsPipeline;
	std::unique_ptr<VulkanCommandPool> commandPool;
	
	std::unique_ptr<vk::raii::Buffer> vertexBuffer;
	std::unique_ptr<vk::raii::DeviceMemory> vertexBufferMemory;
	std::unique_ptr<vk::raii::Buffer> indexBuffer;
	std::unique_ptr<vk::raii::DeviceMemory> indexBufferMemory;
	
	uint32_t mipLevels = {};

	std::unique_ptr<vk::raii::Image> textureImage;
	std::unique_ptr<vk::raii::DeviceMemory> textureImageMemory;
	std::unique_ptr<vk::raii::ImageView> textureImageView;
	std::unique_ptr<vk::raii::Sampler> textureSampler;
	
	std::vector<Vertex> vertices;
	std::vector<uint32_t> indices;

	// hmm combine these two into a class?
	std::vector<std::pair<
		std::unique_ptr<vk::raii::Buffer>,
		std::unique_ptr<vk::raii::DeviceMemory>
	>> uniformBuffers;
	std::vector<void*> uniformBuffersMapped;
	
	std::unique_ptr<vk::raii::DescriptorPool> descriptorPool;
	
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
		))
	{
		msaaSamples = VulkanPhysicalDevice::getMaxUsableSampleCount(physicalDevice);
		std::tie(device, graphicsQueue, presentQueue) = VulkanDevice::create(
			physicalDevice,
			surface,
			deviceExtensions,
			enableValidationLayers
		);
		swapChain = std::make_unique<VulkanSwapChain>(
			app->getScreenSize(),
			physicalDevice,
			*device,
			surface,
			msaaSamples
		);
		graphicsPipeline = std::make_unique<VulkanGraphicsPipeline>(
			physicalDevice,
			*device,
			swapChain->getRenderPass(),
			msaaSamples
		);
		
		{
			auto queueFamilyIndices = VulkanPhysicalDevice::findQueueFamilies(
				physicalDevice,
				surface
			);
			commandPool = std::make_unique<VulkanCommandPool>(
				*device,
				*graphicsQueue,
				vk::CommandPoolCreateInfo()
					.setFlags(vk::CommandPoolCreateFlagBits::eResetCommandBuffer)
					.setQueueFamilyIndex(queueFamilyIndices.graphicsFamily.value())
			);
		}
		
		createTextureImage();
	   
		textureImageView = swapChain->createImageView(
			**textureImage,
			vk::Format::eR8G8B8A8Srgb,
			vk::ImageAspectFlagBits::eColor,
			mipLevels
		);

		textureSampler = std::make_unique<vk::raii::Sampler>(
			*device,
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
		);

		loadModel();
		
		std::tie(vertexBuffer, vertexBufferMemory) 
		= VulkanDeviceMemoryBuffer::makeBufferFromStaged(
			physicalDevice,
			*device,
			*commandPool,
			vertices.data(),
			sizeof(vertices[0]) * vertices.size()
		);

		std::tie(indexBuffer, indexBufferMemory)
		= VulkanDeviceMemoryBuffer::makeBufferFromStaged(
			physicalDevice,
			*device,
			*commandPool,
			indices.data(),
			sizeof(indices[0]) * indices.size()
		);

		for (size_t i = 0; i < maxFramesInFlight; i++) {
			uniformBuffers.push_back(
				VulkanDeviceMemoryBuffer::create(
					physicalDevice,
					*device,
					sizeof(UniformBufferObject),
					vk::BufferUsageFlagBits::eUniformBuffer,
					vk::MemoryPropertyFlagBits::eHostVisible
					| vk::MemoryPropertyFlagBits::eHostCoherent
				)
			);
			uniformBuffersMapped.push_back(std::get<1>(uniformBuffers[i])->mapMemory(
				0,
				sizeof(UniformBufferObject),
				vk::MemoryMapFlags{}
			));
		}

		{
			auto poolSizes = Common::make_array(
				vk::DescriptorPoolSize()
					.setType(vk::DescriptorType::eUniformBuffer)
					.setDescriptorCount(maxFramesInFlight),
				vk::DescriptorPoolSize()
					.setType(vk::DescriptorType::eCombinedImageSampler)
					.setDescriptorCount(maxFramesInFlight)
			);
		
			descriptorPool = std::make_unique<vk::raii::DescriptorPool>(
				*device,
				vk::DescriptorPoolCreateInfo()
					.setMaxSets(maxFramesInFlight)
					//why aren't these two merged into one function?
					.setPoolSizeCount(poolSizes.size())
					.setPPoolSizes(poolSizes.data())
			);
		}

		createDescriptorSets();
		
		initCommandBuffers();
		
		initSyncObjects();
	}

public:
	~VulkanCommon() {
		// vector of unique pointers, can't use `= {}`, gotta use `.clear()`
		commandBuffers.clear();
		descriptorSets.clear();
		imageAvailableSemaphores.clear();
		renderFinishedSemaphores.clear();
		inFlightFences.clear();

		descriptorPool = {};

		uniformBuffers.clear();
		
		indexBufferMemory = {};
		indexBuffer = {};
		vertexBufferMemory = {};
		vertexBuffer = {};

		textureSampler = {};
		textureImageView = {};
		textureImageMemory = {};
		textureImage = {};
		
		graphicsPipeline = {};
		swapChain = {};
		commandPool = {};
		
		device = {};
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
			*device,
			*commandPool,
			srcData,
			bufferSize,
			texSize.x,
			texSize.y,
			mipLevels
		);
	
		generateMipmaps(
			*textureImage,
			vk::Format::eR8G8B8A8Srgb,
			texSize.x,
			texSize.y,
			mipLevels
		);
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
			*device,
			*graphicsQueue,
			(*commandPool)()
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
		device->waitIdle();
		swapChain = std::make_unique<VulkanSwapChain>(
			app->getScreenSize(),
			physicalDevice,
			*device,
			surface,
			msaaSamples
		);
	}

	void initCommandBuffers() {
		// TODO this matches 'VulkanSingleTimeCommand' ctor
		commandBuffers = device->allocateCommandBuffers(
			vk::CommandBufferAllocateInfo()
				.setCommandPool(*(*commandPool)())
				.setLevel(vk::CommandBufferLevel::ePrimary)
				.setCommandBufferCount(maxFramesInFlight)
		);
		// end part that matches
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
				.setRenderPass(*swapChain->getRenderPass())
				.setFramebuffer(*swapChain->framebuffers[imageIndex])
				.setRenderArea(
					vk::Rect2D()
						.setExtent(swapChain->extent)
				)
				.setClearValues(clearValues),
			vk::SubpassContents::eInline
		);

		{
			commandBuffer.bindPipeline(
				vk::PipelineBindPoint::eGraphics,
				**(*graphicsPipeline)()
			);

			commandBuffer.setViewport(
				0,
				Common::make_array(
					vk::Viewport()
						.setWidth(swapChain->extent.width)
						.setHeight(swapChain->extent.height)
						.setMinDepth(0)
						.setMaxDepth(1)
				)
			);

			commandBuffer.setScissor(
				0,
				Common::make_array(
					vk::Rect2D()
						.setExtent(swapChain->extent)
				)
			);

			commandBuffer.bindVertexBuffers(
				0,
				Common::make_array(
					**vertexBuffer
				),
				Common::make_array<vk::DeviceSize>(0)
			);

			commandBuffer.bindIndexBuffer(
				**indexBuffer,
				0,
				vk::IndexType::eUint32
			);

			commandBuffer.bindDescriptorSets(
				vk::PipelineBindPoint::eGraphics,
				**graphicsPipeline->getPipelineLayout(),
				0,
				*descriptorSets[currentFrame],
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
		for (size_t i = 0; i < maxFramesInFlight; i++) {
			imageAvailableSemaphores.push_back(device->createSemaphore({}));
			renderFinishedSemaphores.push_back(device->createSemaphore({}));
			inFlightFences.push_back(device->createFence(
				vk::FenceCreateInfo()
					.setFlags(vk::FenceCreateFlagBits::eSignaled)
			));
		}
	}

	void createDescriptorSets() {
		std::vector<vk::DescriptorSetLayout> layouts(maxFramesInFlight, **graphicsPipeline->getDescriptorSetLayout());
		descriptorSets = device->allocateDescriptorSets(
			vk::DescriptorSetAllocateInfo()
				.setDescriptorPool(**descriptorPool)
				.setDescriptorSetCount(maxFramesInFlight)
				.setSetLayouts(layouts)
		);

		for (size_t i = 0; i < maxFramesInFlight; i++) {
			auto bufferInfo = vk::DescriptorBufferInfo()
				.setBuffer(**std::get<0>(uniformBuffers[i]))
				.setRange(sizeof(UniformBufferObject));
			auto imageInfo = vk::DescriptorImageInfo()
				.setSampler(**textureSampler)
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
			device->updateDescriptorSets(
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
			auto fences = Common::make_array(
				*inFlightFences[currentFrame]
			);
			auto result = device->waitForFences(
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
			std::tie(result, imageIndex) = device->acquireNextImage2KHR(
				vk::AcquireNextImageInfoKHR()
					.setSwapchain(*(*swapChain)())
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

		device->resetFences(
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
		graphicsQueue->submit(
			vk::SubmitInfo()
			.setWaitSemaphores(waitSemaphores)
			.setWaitDstStageMask(waitStages)
			.setCommandBuffers(cmdBufs)
			.setSignalSemaphores(signalSemaphores),
			*inFlightFences[currentFrame]
		);
		
		auto swapChains = Common::make_array(
			*(*swapChain)()
		);
		auto result = presentQueue->presentKHR(
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
		device->waitIdle();
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
