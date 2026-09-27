# Medical Knowledge Base RAG Assistant (DKAI2)

A standalone Retrieval-Augmented Generation (RAG) pipeline built to answer clinical questions using a small corpus of trusted medical guidelines and reference texts. The system indexes 10 authoritative clinical documents across multiple specialties, performs semantic search to retrieve the top 3 relevant passages, and generates factual answers strictly grounded in the retrieved text using an open-source instruction LLM.

---

## Overview

General-purpose language models often generate plausible-sounding medical claims that lack factual grounding. This project demonstrates an evidence-grounded RAG architecture designed to minimize hallucinations through:

1. **Curated Reference Corpus**: 10 clinical guidelines and reference publications from health authorities (WHO, AHA, IHS, DRWF), spanning 430 pages.
2. **Dense Semantic Retrieval**: Passage chunking (800 characters with 150-character overlap) embedded via `BAAI/bge-small-en-v1.5` into ChromaDB and FAISS vector stores.
3. **Constrained Generation**: Prompt-level negative constraints coupled with `Qwen2.5-0.5B-Instruct` in greedy decoding mode to force answers to rely strictly on retrieved excerpts.
4. **Active Refusal Mechanism**: Explicit abstention whenever a query falls outside the indexed knowledge base.

The entire pipeline runs in Google Colab (free-tier T4 GPU or CPU fallback), consuming under 1.5 GB of VRAM.

---

## System Architecture

```text
 10 Medical Reference PDFs (430 pages)
 [Asthma, CKD, Diabetes, Heart Attack, Hypertension, Anemia, Migraine, Pneumonia, Stroke, Tuberculosis]
                               │
                               ▼
                PyPDF Text & Page Extraction
                               │
                               ▼
        Recursive Character Chunking (800 chars, 150 overlap)
        + Deduplication & Noise Filtering (<80 chars)
                               │
                               ▼
         Dense Embeddings (BAAI/bge-small-en-v1.5, 384-d)
                               │
                               ▼
          Vector Database Storage (ChromaDB & FAISS)
                               │
        ┌──────────────────────┴──────────────────────┐
        ▼                                             ▼
   User Query                               Top-3 Chunks Retrieval
   ("What are symptoms of diabetes?")       (Cosine Similarity Search)
        │                                             │
        └──────────────────────┬──────────────────────┘
                               │
                               ▼
             Prompt Construction & Chat Template
             (System constraints, context blocks, page citations)
                               │
                               ▼
       Inference Engine (Qwen2.5-0.5B-Instruct, Greedy Search)
                               │
                               ▼
       Clinical Answer with Document & Page Number Citations
```

---

## Medical Knowledge Corpus

The corpus consists of 10 clinical reference PDFs (total size: ~38.5 MB, 430 extracted pages). Documents were selected from recognized medical societies and public health bodies:

| # | Document File | Source / Authority | Clinical Domain | Focus Area | Pages |
|:-:|:---|:---|:---|:---|:-:|
| 1 | `asthma.pdf` | Global Initiative for Asthma | Pulmonology | Chronic airway inflammation, diagnosis, stepwise management | 32 |
| 2 | `ChronicKidneyDisease.pdf` | Nephrology & Hypertension Division | Nephrology | CKD staging, GFR thresholds, clinical workup | 76 |
| 3 | `diabetes.pdf` | Diabetes Research & Wellness Foundation | Endocrinology | Type 1 vs Type 2 pathology, symptoms, complications | 6 |
| 4 | `heart_attack.pdf` | American Heart Association (AHA) | Cardiology | Acute myocardial infarction signs, sex-specific presentations | 21 |
| 5 | `high-blood-pressure.pdf` | American Heart Association (AHA) | Cardiology | BP classifications (Normal, Elevated, Stage 1 & 2) | 2 |
| 6 | `Iron_Deficiency_and_Iron_Deficiency.pdf` | Clinical Medical Review | Hematology | Anemia etiology, serum ferritin levels, oral/IV iron therapies | 47 |
| 7 | `Migrane.pdf` | International Headache Society (IHS) | Neurology | IHS criteria, aura classification, red flags | 28 |
| 8 | `Pneumonia.pdf` | Department of Pulmonary Medicine | Pulmonology | Community vs hospital-acquired pneumonia, microbiology, signs | 68 |
| 9 | `Stroke.pdf` | American Stroke Association (ASA) | Neurology | Ischemic vs hemorrhagic signs, FAST screening criteria | 2 |
| 10 | `Tuberculosis.pdf` | World Health Organization (WHO) | Infectious Disease | M. tuberculosis transmission, clinical signs, DOTS protocol | 148 |

---

## Pipeline Implementation

### 1. Document Chunking & Filtering
Directly passing full PDF pages creates two problems: dilution of specific clinical facts across hundreds of words, and vector truncation. Conversely, micro-chunks (<200 characters) sever multi-symptom criteria lists.

- **Chunk size**: 800 characters (~140–170 words).
- **Chunk overlap**: 150 characters (ensures symptoms or diagnostic criteria spanning line breaks remain intact).
- **Quality filters**: 
  - Dropped header/footer fragments under 80 characters.
  - Normalized whitespace and stripped duplicate boilerplate chunks across repeating guideline headers.
- **Yield**: 1,380 raw chunks -> 1,303 high-quality chunks indexed.

### 2. Embedding Model
- **Model**: `BAAI/bge-small-en-v1.5`
- **Output dimension**: 384
- **Parameters**: ~33M (disk footprint ~133 MB)
- **Normalization**: L2 normalized embeddings enabled, allowing similarity scoring to map directly to cosine distance.
- **Throughput**: Indexes the entire 1,303-chunk corpus in ~7 seconds on a T4 GPU.

### 3. Vector Storage
- **ChromaDB**: Primary vector store configured with local persistence (`./chroma_medical_db`, collection `medical_knowledge_final`). Maintains chunk text alongside document name and page number metadata.
- **FAISS**: In-memory index used as a lightweight alternative for environments without persistent storage.

### 4. Language Model & Generation Settings
- **Model**: `Qwen/Qwen2.5-0.5B-Instruct`
- **Footprint**: ~490M parameters, running in bfloat16/float16 (~1.1 GB VRAM).
- **Decoding**: Greedy decoding (`do_sample=False`) to enforce deterministic, reproducible extraction and prevent stochastic hallucinations.
- **Context Length**: Configured with a default generation cap of 256 new tokens.

### 5. Grounding Prompt & Safety Guardrails
The system prompt enforces strict constraints to prevent the model from drawing on unverified parametric knowledge:

```text
You are an evidence-grounded Clinical Medical Assistant.
Answer the user's question using ONLY the facts explicitly provided in the medical reference context below.

Strict Safety Rules:
1. Base your answer STRICTLY and EXCLUSIVELY on information explicitly stated in the context.
2. Do NOT extrapolate, speculate, or introduce outside clinical knowledge.
3. If the context does not contain enough information to answer the question, state:
   "The provided medical reference documents do not contain sufficient information to answer this question."
4. Provide structured, accurate, and concise clinical answers.
5. Do not contradict the reference documents.
```

---

## Sample Demonstrations

The notebook evaluates the pipeline on 5 core clinical domain questions plus 1 negative control test.

### Test Case 1: Diabetes (Endocrinology)
- **Question**: *"What are the common symptoms and warning signs of diabetes?"*
- **Retrieved Sources**: `diabetes.pdf` (Pages 2, 3)
- **Generated Answer**:
  > Common Symptoms and Warning Signs of Diabetes:
  > - **Tiredness**: Carbohydrates cannot be converted into energy.
  > - **Urine**: Passing water more frequently as the body removes excess glucose in the urine.
  > - **Thirst**: Drinking large amounts of liquid due to frequent urination.
  > - **Weight Loss**: Glucose is not being used as an energy source, so the body breaks down fat reserves.
  > - **Genital Itching / Thrush**: Glucose in the urine promotes yeast and fungal growth.
  > - **Blurred Vision**: The lens of the eye changes shape as glucose levels fluctuate.

### Test Case 2: Hypertension (Cardiology)
- **Question**: *"What are the blood pressure ranges for Normal, Elevated, and Hypertension Stages 1 and 2?"*
- **Retrieved Sources**: `high-blood-pressure.pdf` (Pages 1, 2)
- **Generated Answer**:
  > - **Normal Blood Pressure**: Systolic less than 120 mm Hg and Diastolic less than 80 mm Hg.
  > - **Elevated Blood Pressure**: Systolic 120–129 mm Hg and Diastolic less than 80 mm Hg.
  > - **Stage 1 Hypertension**: Systolic 130–139 mm Hg or Diastolic 80–89 mm Hg.
  > - **Stage 2 Hypertension**: Systolic 140 mm Hg or higher, or Diastolic 90 mm Hg or higher.

### Test Case 3: Asthma (Pulmonology)
- **Question**: *"What are the primary symptoms of asthma and what common factors can trigger an asthma attack?"*
- **Retrieved Sources**: `asthma.pdf` (Pages 8, 12, 16)
- **Generated Answer**:
  > - **Primary Symptoms**: Recurrent episodes of wheezing, shortness of breath, chest tightness, and coughing, typically worse at night or early morning.
  > - **Common Triggers**: Viral respiratory infections, airborne allergens (dust mites, pollens, animal dander), tobacco smoke, physical exercise, cold air, and strong chemical fumes.

### Test Case 4: Heart Attack (Cardiology)
- **Question**: *"What are the key warning signs of a heart attack, and what differences in symptoms may occur between men and women?"*
- **Retrieved Sources**: `heart_attack.pdf` (Pages 3, 5, 8)
- **Generated Answer**:
  > - **Key Warning Signs**: Chest discomfort (pressure, squeezing, or pain lasting more than a few minutes), discomfort spreading to the back, neck, jaw, or arms, shortness of breath, cold sweat, nausea, or lightheadedness.
  > - **Differences in Women**: While chest pain remains the most common sign for both sexes, women are significantly more likely than men to present with non-chest symptoms, including unexplained shortness of breath, nausea/vomiting, extreme fatigue, and back or jaw pain.

### Test Case 5: Stroke (Neurology)
- **Question**: *"What are the common symptoms of a stroke, and what does the FAST acronym stand for?"*
- **Retrieved Sources**: `Stroke.pdf` (Pages 1, 2)
- **Generated Answer**:
  > - **Common Symptoms**: Sudden numbness or weakness in the face, arm, or leg (especially on one side of the body), sudden confusion, difficulty speaking or understanding speech, vision trouble in one or both eyes, dizziness, and loss of balance.
  > - **F.A.S.T. Acronym**:
  >   - **F (Face Drooping)**: One side of the face droops or is numb.
  >   - **A (Arm Weakness)**: One arm feels weak or numb when raised.
  >   - **S (Speech Difficulty)**: Slurred speech or difficulty repeating a simple sentence.
  >   - **T (Time to Call 911)**: Emergency medical response is needed immediately if any of these signs appear.

### Test Case 6: Negative Control (Hallucination Resistance)
- **Question**: *"What are the surgical steps, donor criteria, and immunosuppression protocols for a lung transplant?"*
- **Retrieved Sources**: Retrieved top-3 fallback passages with low similarity scores from `asthma.pdf` and `Pneumonia.pdf`.
- **Generated Answer**:
  > *"The provided medical reference documents do not contain sufficient information to answer this question."*
- **Outcome**: The model refrained from hallucinating surgical protocols or drawing from general pretraining knowledge, confirming the negative constraint works.

---

## Retrieval Strategy & Limitations

### 1. Retrieval Strategy Analysis

- **Why 800-Character Chunks?**
  Clinical guideline passages present information in structured lists (e.g., diagnostic criteria, stepwise medication tiers). Chunks of 800 characters (~150 words) provide enough context to keep complete symptom clusters together without introducing excessive unrelated text that dilutes the dense vector representation.
- **Overlap Functionality**:
  The 150-character window ensures that boundary sentences—such as exclusion criteria or dosage caveats that occur at the transition between paragraphs—are captured in adjacent chunks rather than cut in half.
- **Top-3 ($k=3$) Context Allocation**:
  Three retrieved chunks total roughly 450–550 words of evidence. For a 0.5B model, keeping the prompt compact prevents attention diffusion ("lost-in-the-middle" issues) while giving the model sufficient factual detail to answer multi-part questions.
- **Dense vs. Keyword Search**:
  Dense embeddings (`bge-small-en-v1.5`) map colloquial queries to formal medical terminology (e.g., matching "high BP" to "hypertension" or "breathing trouble" to "dyspnea"). However, dense retrieval alone can occasionally stumble on exact numerical thresholds (such as specific lab assay cutoffs) where sparse lexical search (BM25) would be more reliable.

### 2. Known Limitations

1. **Static Reference Corpus**:
   The knowledge base contains 10 static documents. Emerging clinical research, updated drug warnings, or regional guideline revisions require manual re-extraction and re-indexing.
2. **Table and Flowchart Degradation**:
   Standard PDF text extractors strip layout coordinates. Complex medication dosing tables and diagnostic decision flowcharts (prominent in `asthma.pdf` and `Tuberculosis.pdf`) lose structural column relationships when flattened into raw text.
3. **Small Parameter Model Constraints**:
   While `Qwen2.5-0.5B-Instruct` handles direct information extraction accurately under strict prompts, its capacity for multi-hop clinical reasoning (e.g., cross-referencing renal clearance stages in CKD with contraindications in diabetes medications) is limited compared to 7B+ parameter models.
4. **No Direct Patient Context**:
   This pipeline functions as a literature retrieval assistant only. It has no integration with electronic health records (EHRs), patient vitals, or laboratory information systems.

---

## Project Structure

```text
.
├── DKAI2.ipynb                # Complete Google Colab notebook with verified execution outputs
├── README.md                  # Project documentation (this file)
└── medical_corpus/            # Folder containing the 10 medical reference PDFs
    ├── asthma.pdf
    ├── ChronicKidneyDisease.pdf
    ├── diabetes.pdf
    ├── heart_attack.pdf
    ├── high-blood-pressure.pdf
    ├── Iron_Deficiency_and_Iron_Deficiency.pdf
    ├── Migrane.pdf
    ├── Pneumonia.pdf
    ├── Stroke.pdf
    └── Tuberculosis.pdf
```

---

## Quickstart & Reproduction

### Running in Google Colab

1. Open [Google Colab](https://colab.research.google.com/) and upload `DKAI2.ipynb`.
2. Set the runtime to **T4 GPU** (`Runtime > Change runtime type > T4 GPU`). The notebook will also execute on CPU if GPU is unavailable.
3. Upload the 10 reference PDFs into a folder named `/content/medical_corpus` or `/content/task2`.
4. Run all cells sequentially (`Runtime > Run all`).
5. Total execution time: ~2–3 minutes from dependency installation through vector indexing and running all test queries.

### Dependencies

```bash
pip install -q \
    "transformers>=4.45.0" \
    "accelerate>=0.34.0" \
    "sentence-transformers>=3.0.0" \
    "langchain>=0.3.0" \
    "langchain-community>=0.3.0" \
    "langchain-core>=0.3.0" \
    "langchain-chroma>=0.1.4" \
    "langchain-huggingface>=0.1.0" \
    "pypdf>=4.3.0" \
    "chromadb>=0.5.0" \
    "faiss-cpu>=1.8.0"
```

---

## Disclaimer

This project is an academic prototype built for educational and technical evaluation purposes. It is **not** a certified medical device and is not approved for clinical decision support or patient care. All medical queries should be addressed by a licensed healthcare professional.
