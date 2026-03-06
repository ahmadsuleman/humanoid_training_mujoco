#Residual Reinforcement Learning for Stable Mocap-to-Humanoid Motion Transfer

## Abstract

Learning physically stable humanoid motion from human demonstrations remains a challenging problem due to kinematic mismatch and dynamic inconsistencies between human motion capture data and robotic embodiments. Motion capture (mocap) datasets provide a rich source of human motion demonstrations, but directly transferring these trajectories to humanoid robots often leads to instability, kinematic mismatch, and motion artifacts due to differences between human and robot morphologies.

In this work, we present a reinforcement learning framework for smooth and stable mocap-to-humanoid motion transfer using the Unitree G1 robot in MuJoCo simulation. Our approach treats mocap trajectories as reference motions and trains a control policy to minimize pose tracking error while enforcing smoothness and physical stability constraints.

To address retargeting noise and dynamic inconsistencies, the policy learns corrective actions that refine reference trajectories through residual control, improving balance, reducing jitter, and mitigating motion artifacts. The reward formulation combines pose tracking accuracy, motion smoothness, contact consistency, and control efficiency, enabling the robot to produce physically plausible movements while remaining faithful to the original mocap demonstrations.

Experiments across diverse motion sequences show that the proposed method significantly improves motion stability and smoothness compared to direct trajectory replay and baseline tracking controllers. These results highlight the potential of reinforcement learning as a scalable approach for bridging human motion datasets and physically consistent humanoid control in simulation and future real-world deployment.

---

## Keywords

- Reinforcement Learning for Control  
- Humanoid Motion Learning  
- Motion Capture Retargeting  
- Embodied Intelligence  
- Continuous Control in Dynamical Systems  
- Humanoid Robotics  
- Physics-Based Simulation

---

## Dataset

This project uses the **KIT Motion-Language Dataset**, a large motion capture dataset containing diverse human activities.

**Dataset features**

- Whole-body motion capture recordings  
- Diverse locomotion and interaction motions  
- Motion sequences paired with semantic descriptions  
- Suitable for motion imitation and embodied AI research  

Dataset link:  
[https://motion-language.github.io/](https://amass.is.tue.mpg.de/)

---

## Project Goal

The objective is to **learn a control policy that reproduces human motion trajectories while preserving physical stability**.

Challenges addressed in this project include:

- Human–robot morphology differences  
- Motion retargeting noise  
- Contact inconsistencies during locomotion  
- Maintaining balance in dynamic movements  

---

## Method Overview

The framework treats motion capture trajectories as **reference motions** and trains a reinforcement learning policy to track them.

### Key components

1. **Motion Retargeting**
   - Human motion capture sequences are mapped to the humanoid skeleton.

2. **Reference Trajectory Generation**
   - Retargeted trajectories provide pose targets for policy learning.

3. **Reinforcement Learning Policy**
   - The policy learns corrective control actions to track the reference motion.

4. **Residual Motion Correction**
   - The policy refines noisy retargeted motions through learned residual control.

---

## Current Development Stage

The current implementation focuses on **legged motion imitation**.

### Stage 1: Lower-Body Control

To simplify the learning problem:

- **Upper body joints are kept fixed**
- The policy controls **lower-body joints only**
- The focus is on learning **stable locomotion patterns**

This allows the RL agent to prioritize:

- balance
- foot contact stability
- smooth stepping motions

### Reference Motion Preparation

For this stage:

- Motion sequences from the KIT dataset have been **fully retargeted**
- Reference trajectories are **ready for reinforcement learning training**
- The simulation environment is prepared for **policy optimization**

---

## Reinforcement Learning Formulation

The RL framework learns a control policy that minimizes the difference between the simulated robot motion and the reference trajectory.

### Objective

The reward function combines:

- pose tracking accuracy  
- motion smoothness  
- contact consistency  
- control efficiency  

This encourages the robot to generate **physically plausible movements while following the reference motion**.

---

## Simulation Environment

- **Physics Engine:** MuJoCo  
- **Robot Model:** Unitree G1 humanoid  
- **Control:** Continuous torque control  
- **Training Framework:** Reinforcement Learning

---

## Future Work

Planned extensions of the project include:

- full-body motion imitation
- upper-body manipulation and gestures
- sim-to-real transfer
- multimodal motion learning using language descriptions

---


## Citation

If you use this work, please cite the associated paper (in preparation).

