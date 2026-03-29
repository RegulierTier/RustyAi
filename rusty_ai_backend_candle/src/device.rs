use candle_core::Device;

/// Where to run Candle ops (CPU or CUDA device index).
#[derive(Clone, Debug)]
pub enum BackendDevice {
    Cpu,
    Cuda(usize),
}

impl BackendDevice {
    /// Resolve to a [`Device`].
    pub fn as_device(&self) -> Device {
        match self {
            BackendDevice::Cpu => Device::Cpu,
            #[cfg(feature = "cuda")]
            BackendDevice::Cuda(i) => Device::new_cuda(*i).unwrap_or(Device::Cpu),
            #[cfg(not(feature = "cuda"))]
            BackendDevice::Cuda(_) => Device::Cpu,
        }
    }
}

/// Prefer CUDA device 0 when the `cuda` feature is enabled and initialization succeeds; otherwise CPU.
#[cfg(feature = "cuda")]
pub fn default_device() -> BackendDevice {
    match Device::new_cuda(0) {
        Ok(_) => BackendDevice::Cuda(0),
        Err(_) => BackendDevice::Cpu,
    }
}

#[cfg(not(feature = "cuda"))]
pub fn default_device() -> BackendDevice {
    BackendDevice::Cpu
}
