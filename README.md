# Multimodal Emotion Recognition with Cross-Attention Transformers

> A CSC413 research project that models six emotions from speech and facial behaviour using Transformer encoders and cross-modal attention.

This project investigates whether emotion can be inferred more effectively from **how people speak and look** than from either signal alone. Using the CMU-MOSEI benchmark, the core experiment learns from acoustic COVAREP features and FACET 4.2 facial features, deliberately excluding text from the primary model to focus on nonverbal affective cues.

The project report is available in [`multimodal_emotion_recognition.pdf`](multimodal_emotion_recognition.pdf).

## Project Highlights

- Built an audio-visual Transformer that models temporal patterns within speech and facial-feature sequences before fusing the modalities with bidirectional cross-attention.
- Created a data workflow for inspecting, filtering, temporally aligning, normalizing, serializing, and batching CMU-MOSEI features.
- Used masks and padded batches to support variable-length utterances without allowing padded frames to influence attention or temporal pooling.
- Compared the primary audio-visual model with an exploratory audio-visual-text extension.
- Evaluated emotion-intensity prediction with a custom weighted agreement metric, weighted F1 for emotion presence, and mean squared error.
- Documented privacy, consent, cultural-bias, surveillance, manipulation, and deployment-risk considerations for affect-recognition systems.

## Problem

Human emotion is communicated through several complementary signals: vocal prosody, facial movements, spoken language, and timing. Single-modality systems can miss cues that another modality captures. This project focuses on the audio and visual streams to isolate nonverbal evidence of emotion and explore how cross-attention can align them over time.

The task is **multi-label, intensity-aware emotion prediction**. For each utterance, the model predicts six emotion intensities—happiness, sadness, anger, surprise, disgust, and fear—on the CMU-MOSEI `0–3` scale. The dataset’s sentiment label is excluded from the primary target.

## Data

The experiments use [CMU-MOSEI](https://multicomp.cs.cmu.edu/resources/cmu-mosei-dataset/), a large multimodal collection of annotated YouTube monologues. Each utterance offers aligned language, audio, and visual information along with sentiment and emotion-intensity annotations.

| Stream | Source features | Shape per time step | Role in primary model |
| --- | --- | ---: | --- |
| Audio | COVAREP prosodic and spectral descriptors | 74 | Captures vocal delivery and acoustic dynamics |
| Video | FACET 4.2 facial descriptors | 35 | Captures facial action units, gaze, and expression signals |
| Text (exploratory) | Timestamped word vectors | 300 | Used only in the AVT comparison notebook |

Class imbalance is a central challenge: common emotions such as happiness appear far more often than rare emotions such as fear. The preprocessing work creates a filtered, more balanced experimental subset, aligns features to utterance-label intervals, handles missing or invalid frames, and serializes the result for efficient notebook experiments.

## Model Architecture

The primary **Audio-Visual Transformer (AV)** has separate encoders for audio and video before information is exchanged across streams.

```text
COVAREP audio features ──> linear projection + positional encoding ─> audio Transformer
                                                                         │
                                                                         ├─ audio attends to video
                                                                         ├─ video attends to audio
                                                                         │
FACET facial features ──> linear projection + positional encoding ─> video Transformer
                                                                         │
                              masked temporal mean pooling <────────────┘
                                                │
                              concatenation → MLP fusion head → 6 emotion intensities
```

Key design choices:

- **Modality-specific Transformer encoders** learn intra-modal temporal dependencies.
- **Bidirectional multi-head cross-attention** lets audio query visual context and visual features query audio context.
- **Residual connections, layer normalization, dropout, and positional encodings** support stable sequence learning.
- **Masked mean pooling** excludes padded frames before modality representations are concatenated.
- A small **MLP fusion head** maps the fused representation to six continuous emotion-intensity outputs.

The exploratory **AVT Transformer** adds a third text encoder and cross-attention connections between all modality pairs. It is included to compare audio-visual learning with a text-inclusive variant; the research report’s central architecture and motivation remain audio-visual.

## Experimental Setup

- **Data split:** 70% train, 15% validation, and 15% test, split by source video before its utterances are added to each partition
- **Optimization:** Adam with mean squared error loss for the Transformer experiments
- **Search space:** learning rate `1e-4`; one or two encoder layers; model widths of 32 or 64; two or four attention heads, depending on the experiment
- **Training:** four epochs per configuration, with the best checkpoint selected by validation weighted F1
- **Hardware support:** PyTorch CUDA automatic mixed precision when a GPU is available

## Recorded Results

The following figures are the test-set outputs saved in [`AV_vs_AVT.ipynb`](AV_vs_AVT.ipynb). They should be read as research-experiment results, not production or state-of-the-art claims.

| Model | Weighted intensity agreement* | Weighted presence F1† | MSE |
| --- | ---: | ---: | ---: |
| Audio + Visual Transformer (AV) | **92.40%** | **44.89%** | **0.1148** |
| Audio + Visual + Text Transformer (AVT, exploratory) | 93.26% | 39.61% | 0.1078 |

\* *Weighted intensity agreement is custom to this project: it averages `max(0, 1 - |target − prediction| / 3)` over emotion intensities.*

† *Presence F1 binarizes each emotion as present when its intensity is greater than zero and uses weighted F1.*

The results illustrate why metric choice matters: the AVT run lowers MSE and improves the continuous-intensity agreement measure, while the AV model yields higher binary emotion-presence F1. The notebook experiments should be reproduced with a standardized evaluation pipeline before drawing broader comparative conclusions.

## My Contributions

As documented in the project report, my contributions included:

- Data cleaning and preprocessing: balancing the data, synchronizing audio and visual features, normalizing inputs, and formatting sequences
- Evaluation and testing: computing metrics, generating analysis artifacts, and interpreting experimental results
- Collaborative report preparation and research communication

This was a team project for the University of Toronto’s CSC413 course. The report credits all collaborators for the overall research, model development, validation, and documentation work.

## Repository Structure

```text
.
├── AV_vs_AVT.ipynb                 # AV Transformer and exploratory AVT comparison
├── MOSEI_cleaning.ipynb             # Dataset inspection, filtering, and balancing workflow
├── MOSEI_data_preprocess.ipynb      # Temporal alignment and serialized dataset preparation
├── text_emotion_detection.ipynb # Text-only baseline exploration
├── multimodal_emotion_recognition.pdf     # Project report
└── README.md
```

## Running the Notebooks

This is a notebook-first research repository. The notebooks were authored for Google Colab and use Colab-style `/content` paths, GPU-aware PyTorch, the CMU Multimodal SDK, and external dataset artifacts. The large source dataset and generated data files are intentionally not committed to the repository.

### Prerequisites

- Python 3.10+
- JupyterLab or Google Colab (recommended)
- A CUDA-capable GPU is recommended for training
- `torch`, `numpy`, `scikit-learn`, `requests`, and the CMU Multimodal SDK
- `aria2` for the notebook data-download workflow

### Suggested workflow

1. Open the notebooks in Google Colab or a local Jupyter environment.
2. Install the dependencies shown in the setup cells, including the CMU Multimodal SDK.
3. Follow `MOSEI_cleaning.ipynb` and `MOSEI_data_preprocess.ipynb` to inspect the data, align features to label intervals, and prepare the serialized dataset artifact.
4. Run `AV_vs_AVT.ipynb` to train and compare the audio-visual and exploratory audio-visual-text Transformer models.
5. Use `text_emotion_detection.ipynb` for the text-only baseline exploration.

Because the notebooks download external artifacts and include exploratory cells, exact runtime and results depend on the available data snapshot, package versions, random seed, and GPU environment. For a production or publication-grade reproduction, the next improvement would be to extract the shared data pipeline and model classes into versioned Python modules, add a dependency lockfile, and centralize the experiment configuration.

## Ethical Considerations

Emotion-recognition systems infer sensitive information from voices and faces. Even when trained on consented, public research data, they can amplify cultural, gender, dialect, or representation biases; be manipulated through deliberate vocal or facial performance; and be misused for surveillance, profiling, hiring, or security decisions.

This project treats predictions as uncertain signals—not objective statements about a person’s internal emotional state. Any real-world deployment would require informed consent, rigorous bias and robustness testing, privacy-preserving data practices, human oversight, and clear limits on high-stakes use.

## Portfolio Summary

This project demonstrates my ability to work across the full research workflow for a multimodal deep-learning problem: understanding a benchmark dataset, engineering synchronized sequential inputs, implementing and evaluating cross-attention Transformer architectures in PyTorch, interpreting competing metrics, and documenting responsible use of affective AI.
