

# **Generative Modeling for Tabular Data Using an Autoencoder–GAN Framework**

## **1. Overview**

This project implements a generative framework for synthesizing realistic tabular data containing both numerical and categorical features.
The architecture combines:

* A **trained Autoencoder (AE)** to learn a stable, continuous latent representation
* A **Generative Adversarial Network (GAN)** trained in that latent space
* Two variants: **Standard GAN** and **Conditional GAN (cGAN)**

This design enables stable GAN training on mixed-type tabular data and allows class-controlled sample generation through conditioning.

High-level architecture training process:

<img width="700" height="200" alt="image" src="https://github.com/user-attachments/assets/a83cb8a4-8e5c-4ca5-bde9-9b943134798e" />

---

## **2. Three-Stage Pipeline**

### **Stage 1 — Autoencoder (AE) Training**

The AE learns a compressed latent representation of mixed-type tabular rows.

<img width="283" height="266" alt="image" src="https://github.com/user-attachments/assets/07f20081-34ef-4bd8-8068-a7225d857659" />

```text
Input Data (numerical + categorical)
        ↓
     [Autoencoder]
        ↓
Latent Embedding (e.g., 32-dim)
```

**Key Details**

* Numerical features → reconstructed using **MSE loss**
* Categorical features → reconstructed using **Cross-Entropy**
* Decoder uses **multiple output heads**, one for each feature type
* AE is trained to minimize:  L_AE = L_MSE(numerical) + L_CE(categorical)
* After convergence, both Encoder and Decoder are **frozen** (they do not update during GAN training)

<img width="827" height="258" alt="image" src="https://github.com/user-attachments/assets/00dd48ce-552e-4f1a-a811-b7b48dc03bac" />


After convergence, both Encoder and Decoder are frozen
(they do not update during GAN training)

---

### **Stage 2 — GAN / cGAN Training (Latent Space)**

GAN training operates entirely within the learned latent space.

#### **Standard GAN (Unconditional)**

```text
Noise → [Generator] → Fake Embedding
Real Sample → [Encoder] → Real Embedding
                     ↓          ↓
                  [Discriminator] → Real / Fake
``` 
Loss Functions (BCE):

Discriminator:
L_D = -[log(D(real)) + log(1 - D(fake))]

Generator:
L_G = -log(D(fake))

These losses are alternated during training — first updating D, then G.
 
#### **Conditional GAN (cGAN)**

```text
Noise + Label → [cGenerator] → Fake Embedding
Real Sample + Label → Encoder → Real Embedding
                              ↓          ↓
                   [cDiscriminator] → Real / Fake (conditioned)
```

Conditioning is implemented by concatenating the one-hot label to both the noise input (Generator) and the embedding input (Discriminator).

Loss functions remain the same, but all inputs are label-conditioned.

The Autoencoder is not updated during GAN/cGAN training.
---

### **Stage 3 — Full Synthetic Data Generation**

#### **Standard GAN**

```text
Noise → Generator → Fake Embedding → AE Decoder → Synthetic Sample
```

#### **Conditional GAN**

```text
Noise + Desired Label → cGenerator → Fake Embedding → Decoder → Synthetic Sample (with the target label)
```

The Decoder reconstructs full tabular rows (both numerical and categorical parts).

---

## **3. Model Variants**

### **Architectural Comparison**

| Component         | Standard GAN             | Conditional GAN (cGAN)            |
| ----------------- | ------------------------ | --------------------------------- |
| **Generator**     | Noise → Embedding        | Noise + One-Hot Label → Embedding |
| **Discriminator** | Embedding → P(real/fake) | Embedding + Label → P(real/fake)  |
| **Output Space**  | Latent (AE) space        | Latent (AE) space                 |

### **Implementation Notes**

* **Generator:**
  ReLU layers, final **Tanh** for normalized latent output
  Trained 3x per Discriminator step (k_steps_g=3)
* **Discriminator:**
  LeakyReLU + Dropout for stability and to prevent overfitting
  Weakened via Dropout(0.5) + LR reduction (×0.25)
*  Result: Balanced training, prevents mode collapse
* **Latent Dimension:**
  Typically 16–64 depending on dataset complexity

---

## **4. Evaluation Methodology**

The framework is evaluated on the **Adult Income Dataset**, a widely used benchmark featuring mixed categorical/numerical inputs.

### **4.1 Detection Metric (Realism)**

*Goal:* measure how distinguishable synthetic samples are from real ones.



**Method:**

* 3 experiments with different seeds → robust results
* Train a Random Forest classifier on a 50/50 mix of real and synthetic samples
* Use a 4-fold evaluation pipeline
* Measure **AUC**


---

### **4.2 Efficacy Metric (Downstream Utility)**

*Goal:* assess how well synthetic data preserves predictive structure.

**Method:**

1. Train a model on **real** training data → test on *real* test data and compute AUC_real
2. Train the same model on **synthetic** data → test on *real* test data and compute AUC_synth 
3. Compute Efficacy Ratio:

```text
Efficacy = AUC_synth / AUC_real
```

**Ideal Result:**
A high ratio (e.g., **0.90–1.00**), showing strong utility of synthetic samples.

---

## **5. Applications & Key Takeaways**

### **General Applications**

* Privacy-preserving synthetic dataset release
* Data augmentation for tabular ML models
* Simulation of realistic data distributions

### **Conditional GAN Advantages**

* **Cybersecurity:**
  Generate rare attack types to improve IDS training
* **Imbalanced Learning:**
  Synthesizing minority-class samples in a controlled way
* **Scenario Simulation:**
  Produce samples with specific attributes on demand

** AE for Anomaly Detection**

The Autoencoder (Stage 1) can be used independently:
High reconstruction error → likely anomaly.


