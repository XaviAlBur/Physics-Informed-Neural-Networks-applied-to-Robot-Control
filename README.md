# Physics-Informed Neural Networks Applied to Robot Modelling and Control

This repository provides an experimental and theoretical implementation of **Physics-Informed Neural Networks (PINNs)** tailored for non-conservative, underactuated, and flexible robotic systems. Based on the framework introduced by Liu et al. (2023), this project incorporates mechanical priors into deep learning architectures using **Lagrangian Neural Networks (LNNs)** and **Hamiltonian Neural Networks (HNNs)**, overcoming key limitations of traditional unconstrained "black-box" models in motor intelligence.

---

## Experiments & Benchmarks

The repository evaluates PINN dynamics learning across three primary robotic scenarios:

1. **Simulated Continuous Planar Soft Robot:** Evaluates non-conservative dynamics learning in continuous soft structures under controlled conditions.
2. **Physical Tendon-Driven Soft Robot:** Benchmarks predictive capability under complex real-world friction and non-colocated tendon forces.
3. **Franka Emika Panda Manipulator:** Trajectory tracking and setpoint regulation validated in PyBullet simulation and physical joint controllers.

---

## Team

* **Xavier Álvarez**
* **Pol Gallego**
* **Jordina Gavaldà**

---

## References

1. **Liu, J., Borja, P., & Della Santina, C. (2023).** *Physics-informed Neural Networks to Model and Control Robots: A Theoretical and Experimental Investigation.* arXiv preprint [arXiv:2305.05375](https://arxiv.org/abs/2305.05375).
2. **Cranmer, M., Greydanus, S., Hoyer, S., Battaglia, P., Spergel, D., & Ho, S. (2020).** *Lagrangian Neural Networks.* ICLR Workshop.
3. **Greydanus, S., Dzamba, M., & Yosinski, J. (2019).** *Hamiltonian Neural Networks.* Advances in Neural Information Processing Systems (NeurIPS), 32.
