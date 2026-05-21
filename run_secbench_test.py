import os
# Suppress OpenMP multi-threading segfaults
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"

import json
from mcp_neural_shield.mcp_classifier_engine import MCPNeuralShield

def main():
    print("=" * 80)
    print("           MCPSecBench Shadow Server & Malicious Client Detection Test       ")
    print("=" * 80)
    
    # 1. Initialize Neural Shield
    print("[1/3] Initializing MCPNeuralShield with model 'shield_model.pt'...")
    registry = MCPNeuralShield(model_path="shield_model.pt", device="cpu")
    print(f"      Neural Shield active on device: {registry.device}")
    
    # 2. Load MCPSecBench Shadow/Malicious tools
    print("\n[2/3] Loading MCPSecBench shadow server/malicious client dataset...")
    with open("secbench_shadow_tests.json", "r") as f:
        shadow_tools = json.load(f)
        
    # 3. Evaluate Detection Performance
    print("\n[3/3] Evaluating shadow/poisoning detection performance using MLP Classifier...")
    
    total_evaluated = 0
    flagged_count = 0
    
    # Keep detail prints clean and print only core hijack cases to prevent verbose console spam
    print("-" * 80)
    print(f"{'Target Tool Name':<28} | {'Attack Category':<30} | {'Status':<15}")
    print("-" * 80)
    
    for tool in shadow_tools:
        total_evaluated += 1
        is_attack = registry.is_attack(tool)
        
        # Determine category label based on description or tool name
        cat = "Shadow Server Hijacking"
        if "cve" in tool["name"] or "exec" in tool["name"] or "sqlite" in tool["name"]:
            cat = "Malicious Client Injection"
        elif "mitm" in tool["name"]:
            cat = "Transport Hijacking (MitM)"
        elif "rebind" in tool["name"]:
            cat = "DNS Rebinding Attack"
        elif "exfiltration" in tool["name"]:
            cat = "Data Exfiltration Probe"
            
        status_str = "BLOCKED" if is_attack else "ALLOWED"
        
        # Print logs for key interesting cases
        if total_evaluated <= 8:
            print(f"{tool['name']:<28} | {cat:<30} | {status_str:<15}")
            
        if is_attack:
            flagged_count += 1
            
    print("-" * 80)
    print(f"... and {total_evaluated - 8} programmatically compiled MCPSecBench test cases evaluated ...")
    print("-" * 80)
    
    # 4. Output Results Summary
    detection_rate = (flagged_count / total_evaluated) * 100
    
    print("\n" + "=" * 80)
    print("                               TEST SUMMARY                                 ")
    print("=" * 80)
    print(f"  - Total MCPSecBench Cases Evaluated : {total_evaluated}")
    print(f"  - Shadow Server / Client Blocked    : {flagged_count} (Expected: {total_evaluated})")
    print(f"  - Detection & Rejection Success Rate: {detection_rate:.1f}%")
    print(f"  - False Pass / Allowed Rate        : {(total_evaluated - flagged_count)/total_evaluated*100:.1f}%")
    print("-" * 80)
    print(f"  OVERALL SECURITY PROTECTION LEVEL   : {'HIGH (100.0% Protected)' if detection_rate == 100.0 else 'CALIBRATING'}")
    print("=" * 80)

if __name__ == "__main__":
    main()
