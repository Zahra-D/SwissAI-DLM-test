# Extension Guides

Complete guides for extending the discrete diffusion library with custom components.

## Available Guides

### Core Components

1.  **[Custom Algorithm](01_custom_algorithm.md)**: Implement a new discrete diffusion method
    *   Inherit from base classes
    *   Override key methods
    *   Register and configure

2.  **[Custom Forward Process](02_custom_forward_process.md)**: Create custom noise patterns
    *   Implement forward diffusion $q(x_t|x_0)$
    *   Integrate with noise schedules
    *   Support custom sampling

3.  **[Custom Noise Schedule](03_custom_noise_schedule.md)**: Define custom time-dependent noise
    *   Implement $\alpha(t)$ and $\alpha'(t)$
    *   Handle boundary conditions
    *   Integrate with algorithms

4.  **[Custom Model](04_custom_model.md)**: Add new backbone architectures
    *   Define model interface
    *   Support time conditioning
    *   Register and configure


## Next Steps

Choose the guide for the component you want to extend and follow along with the complete example. All guides assume you're working from the repository root.

