import os
import json
from mcp_neural_shield.mcp_classifier_engine import MCPNeuralShield

def main():
    print("=" * 70)
    print("      MCP Neural Classifier Poisoning Detection Benchmark      ")
    print("=" * 70)
    
    # Initialize the neural classifier engine
    print("[1/3] Initializing MCPNeuralShield...")
    registry = MCPNeuralShield(model_path="shield_model.pt", device="cpu")
    print(f"      Initialized successfully on: {registry.device}")
    
    # 2. Load datasets
    print("\n[2/3] Loading safe baseline and poisoned test tool schemas...")
    with open("safe_baselines.json", "r") as f:
        safe_tools = json.load(f)
        
    with open("poisoned_tests.json", "r") as f:
        poisoned_tools = json.load(f)
        
    # 3. Run Benchmark
    print("\n[3/3] Running neural attack detection evaluation...")
    
    # Suppress verbose warnings to ensure clean summary logs
    import logging
    logging.getLogger("mcp_neural_shield").setLevel(logging.ERROR)
    
    # Test case A: Identical Tools (should be safe/benign)
    identical_count = 0
    identical_flagged = 0
    for tool in safe_tools:
        identical_count += 1
        if registry.is_attack(tool):
            identical_flagged += 1
            
    # Test case B: Minor Safe Updates (should be safe/benign)
    safe_update_count = 0
    safe_update_flagged = 0
    for tool in safe_tools:
        safe_update_count += 1
        # Create a minor harmless update
        modified_tool = {
            "name": tool["name"],
            "description": tool["description"] + " Updated with standard performance optimizations.",
            "inputSchema": tool["inputSchema"]
        }
        if registry.is_attack(modified_tool):
            safe_update_flagged += 1
            
    # Test case C: Poisoned/Adversarial attacks
    shadow_attack_count = 0
    shadow_attack_flagged = 0
    for shadowed_tool in poisoned_tools:
        shadow_attack_count += 1
        if registry.is_attack(shadowed_tool):
            shadow_attack_flagged += 1
            
    # 4. Output Summary Results
    print("\n" + "=" * 70)
    print("                          BENCHMARK SUMMARY                           ")
    print("=" * 70)
    
    print(f"  A. Identical Baseline Tests:")
    print(f"     - Evaluated: {identical_count}")
    print(f"     - Flagged as Attack: {identical_flagged} (Expected: 0)")
    print(f"     - False Positive Rate: {identical_flagged/identical_count*100:.1f}%")
    
    print(f"\n  B. Safe Minor Update Tests:")
    print(f"     - Evaluated: {safe_update_count}")
    print(f"     - Flagged as Attack: {safe_update_flagged} (Expected: 0)")
    print(f"     - False Positive Rate: {safe_update_flagged/safe_update_count*100:.1f}%")
    
    print(f"\n  C. Poisoned Shadowing Attack Tests:")
    print(f"     - Evaluated: {shadow_attack_count}")
    print(f"     - Flagged as Attack: {shadow_attack_flagged} (Expected: {shadow_attack_count})")
    print(f"     - Detection Success Rate: {shadow_attack_flagged/shadow_attack_count*100:.1f}%")
    
    # Overall Accuracy
    total_evals = identical_count + safe_update_count + shadow_attack_count
    correct_classifications = (identical_count - identical_flagged) + (safe_update_count - safe_update_flagged) + shadow_attack_flagged
    accuracy = correct_classifications / total_evals * 100
    
    print("-" * 70)
    print(f"  OVERALL DETECTOR ACCURACY: {accuracy:.2f}%")
    print("=" * 70)

if __name__ == "__main__":
    main()

