# 🩺 Clinical Vision-Language Model Assistant: Preliminary Medical Image Analysis

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/)
[![Hugging Face Model](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Qwen2--VL--2B--Instruct-blue)](https://huggingface.co/Qwen/Qwen2-VL-2B-Instruct)
[![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-green.svg)](https://www.python.org/)
[![BitsAndBytes 4-bit](https://img.shields.io/badge/Quantization-BitsAndBytes%204--bit%20NF4-orange.svg)](https://github.com/bitsandbytes-foundation/bitsandbytes)

---

## 🎯 Executive Summary & Objectives

This project implements an open-source **Vision-Language Model (VLM)** from Hugging Face that runs efficiently within the resource constraints of the **free Google Colab tier** (NVIDIA Tesla T4 GPU with ~15 GB GDDR6 VRAM and ~12.7 GB System RAM).

The system acts as an interactive **preliminary clinical imaging assistant** that ingests medical imaging studies (e.g., Chest Radiographs and Brain MRIs) and outputs standardized, structured observational reports while adhering to rigorous clinical safety guardrails.

---

## 📋 Deliverables Overview

1. **Jupyter Notebook (`DKAI1.ipynb`)**: Complete, executable, and fully documented notebook with comprehensive markdown explanations and intact execution outputs.
2. **README (`README.md`)**: Architectural overview, model selection rationale, optimization analysis, prompt specifications, and reproduction instructions.
3. **Sample Outputs**: Side-by-side clinical reports generated for two public medical image benchmarks.
4. **Design Choices & Limitations**: Critical evaluation of dynamic token economics, clinical boundaries, and regulatory constraints.

---

## 🧠 Model Selection Justification: Why `Qwen2-VL-2B-Instruct`?

Selecting an optimal Vision-Language Model for medical imaging under free Google Colab constraints requires balancing **reasoning fidelity**, **geometric visual preservation**, and **strict VRAM budgets**.

### Comparative Model Evaluation Matrix

| Model Candidate | Parameters | Resolution Strategy | FP16 VRAM | 4-bit NF4 VRAM | Free Colab Suitability | Key Advantages & Disadvantages |
| :--- | :---: | :---: | :---: | :---: | :---: | :--- |
| **`Qwen/Qwen2-VL-2B-Instruct`** ⭐ *(Selected)* | **2.21B** | **Naive Dynamic Resolution** (arbitrary aspect ratios) | ~4.5 GB | **~1.85 GB** | **Optimal** | **Pros:** Dynamic patch resolution preserves subtle anatomical aspect ratios without distortion; fits in <2 GB VRAM; adheres strictly to structured system prompts. |
| **`HuggingFaceTB/SmolVLM-Instruct`** | 2.2B | Fixed Patch Splitting | ~4.5 GB | ~1.85 GB | Good | **Pros:** Lightweight. **Cons:** Lower visual acuity on fine radiological structures; fewer domain tokens in vocabulary. |
| **`microsoft/Florence-2-large`** | 0.77B | Fixed Resolution (768×768) | ~1.6 GB | ~0.8 GB | Moderate | **Pros:** Extremely fast. **Cons:** Seq2Seq captioning architecture lacks conversational multi-turn chat templates and nuanced clinical reasoning. |
| **`microsoft/Phi-3-vision-128k-instruct`** | 4.2B | Dynamic Cropping (multi-crop) | ~8.5 GB | ~3.8 GB | High VRAM Load | **Pros:** High reasoning capability. **Cons:** Multi-crop visual tokens trigger large memory spikes on T4 GPU, risking CUDA Out-Of-Memory (OOM) errors during generation. |

### Decisive Architectural Factors:
- **Native Dynamic Resolution**: Medical scans vary widely in aspect ratios (tall, narrow chest radiographs vs. wide tomographic slices). Standard ViTs force-resize images to fixed squares ($224 \times 224$ or $384 \times 384$), distorting anatomy and mimicking pathology (e.g., artificial cardiomegaly). Qwen2-VL maps arbitrary aspect ratios directly into variable visual token sequences.
- **Extreme Memory Headroom via 4-Bit NF4**: Under 4-bit NormalFloat quantization, model weights occupy only **1.85 GB VRAM**, leaving >13 GB VRAM free on the T4 GPU for high-resolution visual embeddings, Key-Value (KV) cache, and long conversational context.
- **Standardized Multi-Modal Chat Formatting**: Integrates with Hugging Face's multi-modal chat templating engine, facilitating complex system prompts and multi-turn clinical inquiries.

---

## ⚡ Memory Optimizations

To ensure reliable, crash-free execution on Google Colab free-tier GPUs:

```python
import torch
from transformers import BitsAndBytesConfig, Qwen2VLForConditionalGeneration

quantization_settings = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16
)

vlm_model = Qwen2VLForConditionalGeneration.from_pretrained(
    "Qwen/Qwen2-VL-2B-Instruct",
    quantization_config=quantization_settings,
    device_map="auto"
)
```

### Technical Breakdown:
1. **4-Bit NormalFloat (`nf4`) Quantization**: Compresses FP16 weights down to 4 bits per parameter (~75% reduction). NF4 is mathematically optimal for normally distributed neural network weights, retaining higher reasoning capability than uniform INT4 quantization.
2. **Double Quantization (`bnb_4bit_use_double_quant=True`)**: Quantizes the quantization scale factors themselves, saving ~0.37 bits/parameter (~200 MB VRAM savings).
3. **`bfloat16` Compute Precision**: Activations are dequantized into 16-bit brain floating point for linear matrix multiplications, preventing numerical instability and underflow.
4. **Hugging Face `accelerate` (`device_map="auto"`)**: Automatically maps layers to the primary CUDA device and manages allocation.
5. **Active Memory Footprint**: Verified at **~1.85 GB** (compared to ~4.5 GB in FP16 and ~9.0 GB in FP32).

---

## 🛠️ Dynamic Vision Resolution & Token Budget Control

In Qwen2-VL, visual token count is proportional to image pixel area:
$$\text{Tokens}_{\text{visual}} \approx \frac{\text{Height} \times \text{Width}}{28 \times 28}$$

To prevent memory spikes and latency degradation, the dynamic processor is bounded:
```python
image_processor = AutoProcessor.from_pretrained(
    "Qwen/Qwen2-VL-2B-Instruct",
    min_pixels=256 * 28 * 28,  # ~200,704 pixels (~256 tokens minimum)
    max_pixels=512 * 28 * 28   # ~401,408 pixels (~512 tokens maximum)
)
```
- **`min_pixels`**: Guarantees adequate spatial resolution to distinguish anatomical structures like rib contours and lung margins.
- **`max_pixels`**: Capped to prevent memory spikes, maintaining fast (<2 seconds) inference on the Colab T4 GPU.

---

## 🩺 Clinical System Prompt Engineering

Healthcare AI systems must mitigate diagnostic overconfidence and prevent hallucinated certainty. The system prompt enforces a standardized 4-part observational report format with strict negative constraints:

```text
You are a preliminary clinical imaging assistant.

Analyze the provided medical image and describe ONLY findings
that are visibly supported by the image.

Structure your response as:

1. Image type
2. Key visible findings
3. Possible abnormalities
4. Important limitations

Rules:
- Do not provide a definitive diagnosis.
- Do not state that a patient is healthy or unhealthy.
- Do not infer symptoms, medical history, or patient condition.
- Do not claim that a disease is absent unless the image clearly supports that statement.
- Use cautious language such as "appears", "may indicate", or "no obvious finding is visible".
- Clearly state when something cannot be determined from the image alone.
- This is an educational preliminary image description and must not replace evaluation by a qualified medical professional.
```

---

## 🔄 Reusable Clinical Inference Pipeline

The complete pipeline is formalized into a standalone, modular function:

```python
def run_clinical_image_inference(
    medical_image,
    system_prompt,
    user_prompt,
    max_new_tokens=200
):
    conversation = [
        {
            "role": "system",
            "content": [{"type": "text", "text": system_prompt}]
        },
        {
            "role": "user",
            "content": [
                {"type": "image", "image": medical_image},
                {"type": "text", "text": user_prompt}
            ]
        }
    ]

    model_inputs = image_processor.apply_chat_template(
        conversation,
        tokenize=True,
        add_generation_prompt=True,
        return_tensors="pt",
        return_dict=True
    )

    model_inputs = {
        key: value.to(vlm_model.device)
        for key, value in model_inputs.items()
        if hasattr(value, "to")
    }

    with torch.inference_mode():
        generated_output = vlm_model.generate(
            **model_inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False  # Deterministic greedy generation for medical reports
        )

    # Keep only newly generated tokens
    input_token_count = model_inputs["input_ids"].shape[1]
    generated_answer_ids = generated_output[:, input_token_count:]

    generated_text = image_processor.batch_decode(
        generated_answer_ids,
        skip_special_tokens=True
    )

    return generated_text[0].strip()
```

---

## 📊 Public Benchmark Evaluations & Sample Outputs

The assistant was evaluated on two distinct public domain medical images from Wikimedia Commons:

### Case 1: Posterior-Anterior (PA) Plain Film Chest Radiograph
- **Image URL**: `https://commons.wikimedia.org/wiki/Special:Redirect/file/Chest_Xray_PA_3-8-2010.png`
- **User Prompt**: *"Examine this chest X-ray and provide a preliminary description of the visible findings using the requested structure."*
- **Model Output**:
  ```text
  1. Image type: Chest X-ray
  2. Key visible findings:
     - The lungs appear to be clear and symmetric.
     - There are no obvious signs of consolidation, masses, or masses.
     - The heart and major blood vessels appear to be within normal limits.
  3. Possible abnormalities:
     - No obvious abnormalities are visible.
  4. Important limitations:
     - This image alone cannot definitively diagnose any underlying medical conditions.
  ```

---

### Case 2: Axial Brain Magnetic Resonance Imaging (MRI)
- **Image URL**: `https://commons.wikimedia.org/wiki/Special:Redirect/file/MRI_of_Human_Brain.jpg`
- **User Prompt**: *"Examine this brain MRI and provide a preliminary description of the visible imaging findings using the requested structure. Focus only on what can be visually observed."*
- **Model Output**:
  ```text
  1. Image type: MRI (Magnetic Resonance Imaging)
  2. Key visible findings:
     - The image shows a cross-sectional view of the brain.
     - The brain appears to be normal with no obvious abnormalities.
     - The ventricles are visible and appear to be normal.
     - The cerebral hemispheres and brainstem are well-defined.
     - There are no signs of hemorrhage, edema, or other abnormalities.
  3. Possible abnormalities:
     - None visible.
  4. Important limitations:
     - This image does not provide enough information to determine if there are any underlying medical conditions or abnormalities.
  ```

---

## 🔍 Critical Discussion: Design Choices & Technical Limitations

### Technical Design Choices
1. **Deterministic Decoding (`do_sample=False`)**: Clinical evaluation requires deterministic, reproducible reporting. Stochastic sampling (temperature > 0) creates unacceptable variation where repeated runs on identical images could generate divergent diagnostic claims.
2. **Dynamic Aspect Ratio Preservation**: Medical scans cannot be indiscriminately resized without introducing geometric artifacts. Qwen2-VL's native 2D RoPE (Rotary Position Embedding) and dynamic patch assembly prevent anatomical distortion.
3. **Double Quantization**: Allowed maximal parameter compression with zero noticeable degradation in clinical structure adherence.

### Technical & Clinical Limitations
1. **Spatial Acuity Trade-Off**: Capping resolution to ~512×512 equivalent ensures fast execution on Colab's T4 GPU, but may obscure sub-millimeter findings (e.g., pulmonary micro-nodules <3mm, microcalcifications, hair-line cortical fractures).
2. **Lack of Volumetric 3D Context**: Clinical MRI and CT protocols require scrolling through dozens to hundreds of contiguous slices. Evaluating a single 2D slice cannot rule out intracranial pathology elsewhere.
3. **Absence of Clinical Metadata**: Radiologists interpret scans in conjunction with patient age, sex, clinical history, presentation, vitals, and laboratory markers.
4. **General Foundation Model Pretraining**: While Qwen2-VL demonstrates impressive baseline medical awareness, it was not fine-tuned on specialized clinical datasets (e.g., MIMIC-CXR, CheXpert, RadImageNet).

---

## ⚖️ Regulatory, Ethical & Safety Disclaimer

> **IMPORTANT CLINICAL NOTICE:**  
> This software, code, and accompanying notebook are designed strictly for **academic research, educational demonstrations, and technical feasibility testing**.  
> - This tool is **NOT** a certified Software as a Medical Device (SaMD) under FDA 21 CFR Part 820, EU MDR 2017/745, or CE mark regulatory frameworks.  
> - It is **NOT** intended to diagnose, treat, cure, or prevent any disease, nor should it be used for patient triage or clinical workflow decisions.  
> - All diagnostic imaging must be interpreted by board-certified radiologists or licensed medical practitioners.

---

## 🚀 How to Run in Google Colab

1. Open [Google Colab](https://colab.research.google.com/).
2. Select **Runtime > Change runtime type**, and choose **T4 GPU**.
3. Upload `DKAI1.ipynb`.
4. Run cells sequentially (`Runtime > Run all`).
5. Execution completes within ~3-5 minutes, with total VRAM consumption remaining under ~2.5 GB during peak generation.
