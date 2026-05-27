# Graph Report - .  (2026-05-26)

## Corpus Check
- 12 files · ~20,816 words
- Verdict: corpus is large enough that graph structure adds value.

## Summary
- 267 nodes · 405 edges · 16 communities
- Extraction: 80% EXTRACTED · 20% INFERRED · 0% AMBIGUOUS · INFERRED: 79 edges (avg confidence: 0.65)
- Token cost: 0 input · 0 output

## Community Hubs (Navigation)
- [[_COMMUNITY_RSS Map Processing|RSS Map Processing]]
- [[_COMMUNITY_LS-OFDM Estimation (AST)|LS-OFDM Estimation (AST)]]
- [[_COMMUNITY_PINN Model Architecture|PINN Model Architecture]]
- [[_COMMUNITY_RSS Mapping & Transfer|RSS Mapping & Transfer]]
- [[_COMMUNITY_Physics-Informed Training|Physics-Informed Training]]
- [[_COMMUNITY_PINN Cross-Attention|PINN Cross-Attention]]
- [[_COMMUNITY_Data Pipeline|Data Pipeline]]
- [[_COMMUNITY_Evaluation & Testing|Evaluation & Testing]]
- [[_COMMUNITY_Ray-Tracing Channel Synthesis|Ray-Tracing Channel Synthesis]]
- [[_COMMUNITY_Fine-Tuning Rationale|Fine-Tuning Rationale]]
- [[_COMMUNITY_RSS Color Mapping|RSS Color Mapping]]
- [[_COMMUNITY_Initial LS Estimation|Initial LS Estimation]]
- [[_COMMUNITY_Physics Loss Components|Physics Loss Components]]
- [[_COMMUNITY_Canyon RSS Heatmap|Canyon RSS Heatmap]]
- [[_COMMUNITY_Urban 15GHz Propagation|Urban 15GHz Propagation]]
- [[_COMMUNITY_Urban 8GHz Propagation|Urban 8GHz Propagation]]

## God Nodes (most connected - your core abstractions)
1. `RSSMapProcessor` - 53 edges
2. `main_train()` - 13 edges
3. `GlobalNormalizedDataset` - 12 edges
4. `GlobalNormalizedDataset` - 12 edges
5. `PhysicsInformedLoss` - 10 edges
6. `PhysicsInformedLoss` - 10 edges
7. `LSOFDMChannelEstimator` - 9 edges
8. `ImprovedRSSColorMapper` - 8 edges
9. `ImprovedRSSColorMapper` - 8 edges
10. `TransferLearningExperiment` - 7 edges

## Surprising Connections (you probably didn't know these)
- `NMSE (Normalized Mean Square Error) Metric` --semantically_similar_to--> `NMSE Loss (L_NMSE reconstruction term)`  [INFERRED] [semantically similar]
  README.md → Model.py
- `Positional encoding for transformer` --uses--> `RSSMapProcessor`  [INFERRED]
  Model.py → find_in_map.py
- `Transformer decoder for processing channel features - UNCHANGED` --uses--> `RSSMapProcessor`  [INFERRED]
  Model.py → find_in_map.py
- `Args:             memory: Encoded features from U-Net encoder (batch, seq_len, d` --uses--> `RSSMapProcessor`  [INFERRED]
  Model.py → find_in_map.py
- `Enhanced cross-attention with better feature fusion and residual connection` --uses--> `RSSMapProcessor`  [INFERRED]
  Model.py → find_in_map.py

## Hyperedges (group relationships)
- **PINN Inference Pipeline: LS Estimate + RSS Map -> Refined Channel** — init_LSOFDMChannelEstimator, find_in_map_RSSMapProcessor, model_ImprovedPhysicsInformedUNet [EXTRACTED 0.97]
- **Physics-Informed Loss: NMSE + Physical Power Regularization** — model_PhysicsInformedLoss, model_NMSE_loss, model_power_loss [EXTRACTED 1.00]
- **Ground-Truth Channel Tensor Construction from Ray-Tracing** — make_correct_build_channel_tensor, make_correct_raised_cosine_pulse, make_correct_array_response_UPA, make_correct_make_complex_gain [EXTRACTED 1.00]

## Communities (16 total, 0 thin omitted)

### Community 0 - "RSS Map Processing"
Cohesion: 0.07
Nodes (24): main(), Visualize all user locations on the map.          Args:             user_locatio, Process a dataset of user locations.          Args:             user_data_path:, Convert real-world coordinates to pixel coordinates.          Args:, Crop the image around a user's location.              Args:             real_x:, Initialize the RSS map processor.          Args:             image_path: Path to, RSSMapProcessor, create_datasets (+16 more)

### Community 1 - "LS-OFDM Estimation (AST)"
Cohesion: 0.10
Nodes (16): create_ls_ofdm_estimates(), InitialChannelEstimator, LSOFDMChannelEstimator, LS OFDM Channel Estimation          This implements the standard LS estimation i, LS OFDM estimation with frequency domain smoothing.         This exploits the fa, MMSE OFDM channel estimation (better than LS but requires channel statistics)., Initialize LS OFDM channel estimator.                  Args:             N_tap:, Create LS OFDM channel estimates.          Args:         true_channels_file: Pat (+8 more)

### Community 2 - "PINN Model Architecture"
Cohesion: 0.10
Nodes (11): EnhancedCrossAttention, ImprovedPhysicsInformedUNet, PositionalEncoding, Positional encoding for transformer, Transformer decoder for processing channel features - UNCHANGED, Args:             memory: Encoded features from U-Net encoder (batch, seq_len, d, Enhanced cross-attention with better feature fusion and residual connection, Enhanced U-Net block with residual connections (+3 more)

### Community 3 - "RSS Mapping & Transfer"
Cohesion: 0.10
Nodes (22): RSSMapProcessor, crop_around_user (RSS map crop), real_to_pixel (coordinate conversion), visualize_users, TransferLearningExperiment, EnhancedCrossAttention, GlobalNormalizedDataset, ImprovedPhysicsInformedUNet (+14 more)

### Community 4 - "Physics-Informed Training"
Cohesion: 0.12
Nodes (16): fine_tune_model (few-shot fine-tuning loop), NMSE Loss (L_NMSE reconstruction term), PhysicsInformedLoss, load_checkpoint, PhysicsInformedLoss, Physical Power Loss (L_physical RSS correlation), Combined loss function: NMSE + alpha * MSE(RSS, channel_power), Convert real/imag tensor back to complex (+8 more)

### Community 5 - "PINN Cross-Attention"
Cohesion: 0.12
Nodes (9): EnhancedCrossAttention, PositionalEncoding, Positional encoding for transformer, Transformer decoder for processing channel features - UNCHANGED, Args:             memory: Encoded features from U-Net encoder (batch, seq_len, d, Enhanced cross-attention with better feature fusion and residual connection, Enhanced U-Net block with residual connections, ResidualUNetBlock (+1 more)

### Community 6 - "Data Pipeline"
Cohesion: 0.11
Nodes (12): Dataset, create_datasets(), GlobalNormalizedDataset, Maps RSS colorbar colors to dBm values, Convert RGB RSS map to dBm values, Normalize dBm values to [-1, 1] for neural network, Dataset with global normalization applied before train/test split, Get normalization parameters for saving (+4 more)

### Community 7 - "Evaluation & Testing"
Cohesion: 0.18
Nodes (12): evaluate_test_set, Evaluate model on test set with global normalization, evaluate_test_set(), ImprovedPhysicsInformedUNet, load_checkpoint(), Improved U-Net with transformer decoder, enhanced attention, and residual connec, Save training checkpoint, Load training checkpoint if it exists          Returns:         start_epoch, tra (+4 more)

### Community 8 - "Ray-Tracing Channel Synthesis"
Cohesion: 0.26
Nodes (11): array_response_UPA(), build_channel_tensor(), main(), make_complex_gain(), _parse_args(), raised_cosine_pulse(), Compute the raised cosine pulse response for a given delay.     Handles cases wh, Computes 2D UPA response using Kronecker structure. (+3 more)

### Community 9 - "Fine-Tuning Rationale"
Cohesion: 0.24
Nodes (5): Experiment class for fine-tuning a model trained on 15 GHz data to work with oth, Evaluate a model on the validation set.                  Args:             model, Run the complete transfer learning experiment using validation set for evaluatio, Fine-tune the pretrained model on a subset of 8 GHz data.                  Args:, TransferLearningExperiment

### Community 10 - "RSS Color Mapping"
Cohesion: 0.20
Nodes (6): ImprovedRSSColorMapper, Improved RSS color to dBm mapper using actual colormap lookup, Build a lookup table from colors to dBm values, Convert RGB to dBm using nearest neighbor in color space, Normalize dBm values to [-1, 1] using actual data range, Create a reference colorbar for visualization

### Community 11 - "Initial LS Estimation"
Cohesion: 0.20
Nodes (10): InitialChannelEstimator, LSOFDMChannelEstimator, create_ls_ofdm_estimates, Wireless Insite Ray-Tracing CSV Input, array_response_UPA (UPA steering vector), build_channel_tensor (ray-tracing to MIMO channel), make_complex_gain (path complex gain), raised_cosine_pulse (pulse shaping filter) (+2 more)

### Community 12 - "Physics Loss Components"
Cohesion: 0.33
Nodes (5): PhysicsInformedLoss, Combined loss function: NMSE + alpha * MSE(RSS, channel_power), Convert real/imag tensor back to complex, Calculate Normalized Mean Square Error for complex channels, Calculate channel power across spatial dimensions

### Community 13 - "Canyon RSS Heatmap"
Cohesion: 0.39
Nodes (8): Building Footprints (Urban Blockage), Channel Estimation Dataset, Urban Canyon Environment, Carrier Frequency (50 or 1.5 GHz), RF Signal Strength Heatmap - Canyon Urban Environment at 50/1.5GHz, High Signal Strength Region (Red/Orange - High Power), Low Signal Strength Region (Blue/Purple - Low Power), RF Signal Propagation Path

### Community 14 - "Urban 15GHz Propagation"
Cohesion: 0.36
Nodes (8): Base Station Location (Red Dot), Channel Estimation Dataset, 1.5 GHz Carrier Frequency, Multipath Propagation Effects, Physics-Informed Neural Network (PINN), Radio Propagation Heatmap at 1.5 GHz (50m range), Received Signal Strength / Path Loss, Urban Environment Layout

### Community 15 - "Urban 8GHz Propagation"
Cohesion: 0.38
Nodes (7): Base Station / Transmitter Location, Channel Estimation Dataset Sample, Path Loss and Shadowing Effects, Physics-Informed Neural Network (PINN), Radio Propagation Heatmap at 8GHz (50th sample), Received Signal Strength Indicator (RSSI), Urban Environment Layout

## Knowledge Gaps
- **22 isolated node(s):** `visualize_users`, `ResidualUNetBlock`, `PositionalEncoding`, `ImprovedRSSColorMapper`, `InitialChannelEstimator` (+17 more)
  These have ≤1 connection - possible missing edges or undocumented components.

## Suggested Questions
_Questions this graph is uniquely positioned to answer:_

- **Why does `RSSMapProcessor` connect `RSS Map Processing` to `PINN Model Architecture`, `Physics-Informed Training`, `PINN Cross-Attention`, `Data Pipeline`, `Evaluation & Testing`, `Fine-Tuning Rationale`, `RSS Color Mapping`, `Physics Loss Components`?**
  _High betweenness centrality (0.338) - this node is a cross-community bridge._
- **Why does `main_train()` connect `Evaluation & Testing` to `RSS Map Processing`, `PINN Model Architecture`, `RSS Mapping & Transfer`, `Physics-Informed Training`, `Data Pipeline`, `Initial LS Estimation`?**
  _High betweenness centrality (0.169) - this node is a cross-community bridge._
- **Why does `PhysicsInformedLoss` connect `Physics Loss Components` to `RSS Map Processing`, `Fine-Tuning Rationale`, `PINN Cross-Attention`, `Evaluation & Testing`?**
  _High betweenness centrality (0.057) - this node is a cross-community bridge._
- **Are the 45 inferred relationships involving `RSSMapProcessor` (e.g. with `ImprovedRSSColorMapper` and `RSSColorMapper`) actually correct?**
  _`RSSMapProcessor` has 45 INFERRED edges - model-reasoned connections that need verification._
- **Are the 6 inferred relationships involving `main_train()` (e.g. with `RSSMapProcessor` and `ImprovedPhysicsInformedUNet`) actually correct?**
  _`main_train()` has 6 INFERRED edges - model-reasoned connections that need verification._
- **Are the 2 inferred relationships involving `GlobalNormalizedDataset` (e.g. with `RSSMapProcessor` and `._setup_datasets()`) actually correct?**
  _`GlobalNormalizedDataset` has 2 INFERRED edges - model-reasoned connections that need verification._
- **Are the 2 inferred relationships involving `GlobalNormalizedDataset` (e.g. with `RSSMapProcessor` and `._setup_datasets()`) actually correct?**
  _`GlobalNormalizedDataset` has 2 INFERRED edges - model-reasoned connections that need verification._