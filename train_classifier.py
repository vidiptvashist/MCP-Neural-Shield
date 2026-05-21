#!/usr/bin/env python
import os
import json
import time
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sentence_transformers import SentenceTransformer

# Try importing scikit-learn for metrics and comparison classifiers, fallback gracefully if not installed
try:
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import precision_recall_fscore_support, classification_report
    from sklearn.svm import SVC
    from sklearn.neural_network import MLPClassifier as SKMLP
    HAS_SKLEARN = True
except ImportError:
    HAS_SKLEARN = False

# Suppress multi-threading conflicts on Apple Silicon
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"

# -----------------------------------------------------------------------------
# 1. PyTorch Model Architecture
# -----------------------------------------------------------------------------
class ToolMLPClassifier(nn.Module):
    """
    Lightweight Multi-Layer Perceptron (MLP) to classify tool descriptions
    as Safe (0) or Poisoned/Attack (1).
    """
    def __init__(self, input_dim: int = 384, hidden_dim: int = 64):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 1)  # Outputs raw logits for Binary Cross-Entropy
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x)

# -----------------------------------------------------------------------------
# 2. Tool Serialization matching MCPSemanticRegistry
# -----------------------------------------------------------------------------
def serialize_tool(tool_schema: dict) -> str:
    """
    Serializes tool metadata (name, description, input properties)
    into a structured string for semantic embedding.
    """
    name = tool_schema.get("name", "")
    description = tool_schema.get("description", "")
    input_schema = tool_schema.get("inputSchema", {})

    properties = input_schema.get("properties", {}) if isinstance(input_schema, dict) else {}
    required = input_schema.get("required", []) if isinstance(input_schema, dict) else []

    props_str = (
        ", ".join(f"{k} ({v.get('type', 'any')})" for k, v in properties.items())
        if properties
        else "none"
    )
    req_str = ", ".join(required) if required else "none"

    return f"Tool Name: {name}. Description: {description}. Input properties: {props_str}. Required inputs: {req_str}."

# -----------------------------------------------------------------------------
# 3. Data Augmentation: Diverse Naturally-Phrased Safe Tools
# -----------------------------------------------------------------------------
def _generate_diverse_safe_augmentation():
    """
    Generates a large corpus of diverse, naturally-worded safe tool descriptions 
    to augment the training set. This eliminates the bias toward rigid synthetic 
    patterns like 'Standard X utility tool...' and teaches the model that clean 
    tools come in many natural phrasing styles.
    
    Generates ~1,500+ samples to reach parity with existing synthetic safe data,
    ensuring the MLP learns from description semantics rather than surface patterns.
    """
    import random as _rng
    _rng.seed(2026)

    # --- PART 1: Core realistic tool descriptions (61 unique descriptions) ---
    realistic_tools = [
        # Filesystem / File Management
        {"name": "read_logs", "description": "Read application log files securely from the local filesystem sandboxed path."},
        {"name": "read_file", "description": "Reads the contents of a file at the specified path and returns its text."},
        {"name": "write_file", "description": "Writes content to a file, creating it if it doesn't exist."},
        {"name": "list_directory", "description": "Lists all files and subdirectories in the given directory path."},
        {"name": "search_files", "description": "Search for files matching a pattern within a directory tree."},
        {"name": "move_file", "description": "Moves or renames a file from source to destination path."},
        {"name": "get_file_info", "description": "Returns metadata about a file including size, modification date, and permissions."},
        {"name": "create_directory", "description": "Creates a new directory at the specified path, including parent directories."},
        {"name": "tail_log", "description": "Returns the last N lines from a log file for quick debugging."},
        {"name": "watch_directory", "description": "Monitors a directory for file changes and reports additions or deletions."},
        {"name": "copy_file", "description": "Copies a file from one location to another within the workspace."},
        {"name": "delete_file", "description": "Removes a file from disk after confirming the path is within the workspace."},
        {"name": "read_text", "description": "Opens and reads a text file, returning its contents as a UTF-8 string."},
        {"name": "save_output", "description": "Saves the given text output to a file at the specified path."},
        # Git / Version Control
        {"name": "git_status", "description": "Shows the working tree status of the current Git repository."},
        {"name": "git_diff", "description": "Displays differences between commits, the working tree, and the index."},
        {"name": "git_log", "description": "Shows commit logs with author, date, and message for the repository."},
        {"name": "git_commit", "description": "Records changes to the repository with a descriptive commit message."},
        {"name": "git_branch", "description": "Lists, creates, or deletes branches in the Git repository."},
        {"name": "git_clone", "description": "Clones a remote repository into a new local directory."},
        {"name": "git_push", "description": "Pushes local commits to the configured remote repository."},
        {"name": "git_pull", "description": "Fetches and merges changes from the remote repository."},
        # Database
        {"name": "query_database", "description": "Executes a read-only SQL query against the configured database and returns results."},
        {"name": "list_tables", "description": "Lists all tables in the connected database schema."},
        {"name": "describe_table", "description": "Returns the column names, types, and constraints for a database table."},
        {"name": "insert_record", "description": "Inserts a new record into the specified database table."},
        {"name": "run_migration", "description": "Applies pending database migrations in order."},
        {"name": "count_rows", "description": "Returns the number of rows in a given database table."},
        # HTTP / API
        {"name": "fetch_url", "description": "Fetches the content of a URL and returns the response body as text."},
        {"name": "http_get", "description": "Makes an HTTP GET request to the specified endpoint and returns the response."},
        {"name": "http_post", "description": "Sends an HTTP POST request with a JSON body to the specified URL."},
        {"name": "check_endpoint", "description": "Checks if an API endpoint is reachable and returns its HTTP status code."},
        {"name": "download_file", "description": "Downloads a file from a URL and saves it to the local filesystem."},
        {"name": "upload_artifact", "description": "Uploads a build artifact to the configured storage backend."},
        # Search
        {"name": "web_search", "description": "Performs a web search and returns relevant results with titles and snippets."},
        {"name": "search_code", "description": "Searches through the codebase for files containing the specified text pattern."},
        {"name": "semantic_search", "description": "Performs a semantic similarity search against indexed documents."},
        {"name": "find_references", "description": "Finds all references to a symbol across the project source files."},
        {"name": "grep_workspace", "description": "Runs a grep-like search across all files in the workspace directory."},
        # Development Tools
        {"name": "run_tests", "description": "Executes the project's test suite and reports pass/fail results."},
        {"name": "lint_code", "description": "Runs the linter on source files and reports style and quality issues."},
        {"name": "format_code", "description": "Automatically formats source code according to the project's style guide."},
        {"name": "build_project", "description": "Compiles the project and produces the output artifacts."},
        {"name": "debug_process", "description": "Attaches to a running process for interactive debugging."},
        {"name": "analyze_deps", "description": "Analyzes project dependencies and reports outdated or vulnerable packages."},
        {"name": "generate_docs", "description": "Generates API documentation from source code annotations and docstrings."},
        {"name": "profile_code", "description": "Profiles code execution to identify performance bottlenecks."},
        {"name": "check_types", "description": "Runs static type checking on the codebase and reports type errors."},
        # System / DevOps
        {"name": "check_memory", "description": "Returns current system memory usage statistics including free and used RAM."},
        {"name": "check_disk", "description": "Reports disk space usage for mounted filesystems."},
        {"name": "list_processes", "description": "Lists currently running processes with their PID, CPU, and memory usage."},
        {"name": "get_env_var", "description": "Retrieves the value of an environment variable by name."},
        {"name": "restart_service", "description": "Restarts a managed system service by name."},
        {"name": "view_logs", "description": "Displays recent entries from the system or application log."},
        {"name": "deploy_app", "description": "Deploys the application to the specified environment using the CI/CD pipeline."},
        {"name": "health_check", "description": "Runs a health check against the application and reports component status."},
        {"name": "get_uptime", "description": "Returns the system uptime and load average statistics."},
        {"name": "check_ports", "description": "Lists open network ports and the processes listening on them."},
        # Data Processing
        {"name": "parse_json", "description": "Parses a JSON string and returns the structured data object."},
        {"name": "parse_csv", "description": "Reads a CSV file and returns rows as a list of dictionaries."},
        {"name": "convert_format", "description": "Converts data between formats such as JSON, YAML, TOML, and XML."},
        {"name": "validate_schema", "description": "Validates a data structure against a JSON Schema definition."},
        {"name": "calculate_hash", "description": "Computes the SHA-256 hash of a file or string for integrity verification."},
        {"name": "compress_file", "description": "Compresses a file or directory into a zip or tar.gz archive."},
        {"name": "extract_archive", "description": "Extracts the contents of a compressed archive to a target directory."},
        {"name": "sort_data", "description": "Sorts a dataset by the specified column or key in ascending or descending order."},
        {"name": "filter_records", "description": "Filters records from a dataset based on the provided query criteria."},
        {"name": "merge_files", "description": "Merges multiple files into a single output file."},
        # Communication
        {"name": "send_email", "description": "Sends an email message to specified recipients via the configured SMTP server."},
        {"name": "send_notification", "description": "Sends a push notification to the user's configured notification channel."},
        {"name": "post_message", "description": "Posts a message to the specified Slack channel or team chat."},
        {"name": "create_ticket", "description": "Creates a new issue or ticket in the configured project tracker."},
        # Math / Utility
        {"name": "calculate", "description": "Evaluates a mathematical expression and returns the numeric result."},
        {"name": "generate_uuid", "description": "Generates a new UUID v4 string for unique identification."},
        {"name": "convert_timezone", "description": "Converts a timestamp from one timezone to another."},
        {"name": "regex_match", "description": "Tests a string against a regular expression pattern and returns matches."},
        {"name": "encode_base64", "description": "Encodes the input string to Base64 format."},
        {"name": "decode_base64", "description": "Decodes a Base64 encoded string back to plain text."},
        {"name": "random_string", "description": "Generates a random alphanumeric string of the specified length."},
        {"name": "timestamp_now", "description": "Returns the current date and time as an ISO 8601 formatted string."},
    ]

    # --- PART 2: Diverse description templates (20 phrasings) ---
    desc_templates = [
        "Provides a simple interface to {action} within the project workspace.",
        "Handles {action} operations efficiently with proper error handling.",
        "A utility that {action} and returns structured results.",
        "Performs {action} in a sandboxed environment with access controls.",
        "Safely {action} while respecting configured rate limits and permissions.",
        "Enables users to {action} through a clean programmatic API.",
        "Manages {action} tasks with built-in retry logic and timeout handling.",
        "Supports {action} across multiple platforms and environments.",
        "Runs {action} as a background task and reports progress.",
        "A lightweight tool for {action} with minimal dependencies.",
        "Coordinates {action} workflows with configurable parameters.",
        "Automates {action} according to the project's conventions.",
        "Implements {action} following security best practices.",
        "Executes {action} and logs the results for auditing.",
        "Orchestrates {action} pipelines with proper cleanup on failure.",
        "A helper that {action} and caches results for performance.",
        "Wraps {action} in a transaction-safe execution context.",
        "Streamlines {action} by providing sensible defaults.",
        "Facilitates {action} with comprehensive input validation.",
        "Delegates {action} to the appropriate backend service.",
    ]

    # --- PART 3: Actions covering many domains (50 actions) ---
    actions = [
        "read and display file contents", "write data to disk", "search through directories",
        "query database tables", "fetch web resources", "run automated tests",
        "manage version control operations", "monitor system health",
        "process and transform data formats", "manage container deployments",
        "generate reports from metrics", "validate configuration files",
        "manage user permissions", "cache frequently accessed data",
        "schedule background tasks", "analyze code quality metrics",
        "manage API authentication tokens", "aggregate log entries",
        "perform batch data imports", "export results to spreadsheets",
        "manage environment configurations", "synchronize remote repositories",
        "optimize database queries", "manage session state",
        "render markdown to HTML", "manage webhook subscriptions",
        "parse structured text input", "format output for display",
        "check network connectivity", "manage encryption keys",
        "scan for dependency vulnerabilities", "index project files",
        "compile source code modules", "package application releases",
        "rotate application log files", "archive completed tasks",
        "inspect container images", "manage DNS records",
        "validate JSON schemas", "compute file checksums",
        "generate random test data", "clean temporary files",
        "profile memory allocations", "measure response latency",
        "list available plugins", "register event handlers",
        "update package dependencies", "verify SSL certificates",
        "backup database snapshots", "restore previous configurations",
    ]

    tool_name_prefixes = [
        "file", "data", "dir", "db", "url", "test", "vcs", "health",
        "transform", "container", "report", "config", "perm", "cache",
        "task", "code", "auth", "log", "batch", "export", "env", "repo",
        "query", "session", "render", "webhook", "parser", "formatter",
        "network", "crypto", "scanner", "indexer", "compiler", "packager",
        "rotator", "archiver", "inspector", "dns", "validator", "hasher",
        "generator", "cleaner", "profiler", "latency", "plugin", "event",
        "updater", "ssl", "backup", "restore",
    ]

    # --- PART 4: Varied input schemas to prevent shortcut learning ---
    input_schemas = [
        {"type": "object", "properties": {"path": {"type": "string"}}, "required": ["path"]},
        {"type": "object", "properties": {"query": {"type": "string"}}, "required": ["query"]},
        {"type": "object", "properties": {"input": {"type": "string", "description": "Primary input parameter."}}, "required": ["input"]},
        {"type": "object", "properties": {"target": {"type": "string"}, "options": {"type": "object"}}, "required": ["target"]},
        {"type": "object", "properties": {"name": {"type": "string"}, "value": {"type": "string"}}, "required": ["name"]},
        {"type": "object", "properties": {"source": {"type": "string"}, "destination": {"type": "string"}}, "required": ["source", "destination"]},
        {"type": "object", "properties": {"url": {"type": "string"}}, "required": ["url"]},
        {"type": "object", "properties": {"command": {"type": "string"}, "args": {"type": "array"}}, "required": ["command"]},
        {"type": "object", "properties": {"lines": {"type": "integer"}}, "required": []},
        {"type": "object", "properties": {"filename": {"type": "string"}, "content": {"type": "string"}}, "required": ["filename"]},
        {"type": "object", "properties": {"expression": {"type": "string"}}, "required": ["expression"]},
        {"type": "object", "properties": {"pattern": {"type": "string"}, "directory": {"type": "string"}}, "required": ["pattern"]},
    ]

    augmented = []

    # 1. Add all realistic tools with VARIED input schemas (4x oversampling)
    #    This is critical: each tool appears multiple times with different schemas
    #    so the model can't use schema format as a discrimination shortcut.
    for tool in realistic_tools:
        for schema_idx in range(4):
            schema = _rng.choice(input_schemas)
            augmented.append({
                "name": tool["name"] if schema_idx == 0 else f"{tool['name']}_{schema_idx}",
                "description": tool["description"],
                "inputSchema": schema
            })

    # 2. Generate template × action combinations with varied schemas (1000+ samples)
    for i, action in enumerate(actions):
        for j, template in enumerate(desc_templates):
            desc = template.format(action=action)
            prefix = tool_name_prefixes[i % len(tool_name_prefixes)]
            suffix = f"_{j}" if j > 0 else ""
            augmented.append({
                "name": f"{prefix}_tool{suffix}",
                "description": desc,
                "inputSchema": input_schemas[(i + j) % len(input_schemas)]
            })

    print(f"    - Generated {len(augmented)} diverse naturally-phrased augmentation samples.")
    return augmented


# -----------------------------------------------------------------------------
# 4. Main Training Routine
# -----------------------------------------------------------------------------
def main():
    print("=" * 80)
    print("           mcp-vector-shield: Neural Network Classifier Training            ")
    print("=" * 80)

    # A. Check and select device
    if torch.cuda.is_available():
        device = "cuda"
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        device = "mps"
    else:
        device = "cpu"
    print(f"[*] Training pipeline hardware device selected: {device.upper()}")

    # B. Load Datasets
    print("\n[*] Loading localized datasets from disk...")
    safe_samples = []
    poisoned_samples = []

    # Safe Baselines (Label 0)
    if os.path.exists("safe_baselines.json"):
        with open("safe_baselines.json", "r") as f:
            safe_baselines = json.load(f)
            safe_samples.extend(safe_baselines)
            print(f"    - Loaded {len(safe_baselines)} safe tools from safe_baselines.json")

    if os.path.exists("massive_safe_baselines.json"):
        with open("massive_safe_baselines.json", "r") as f:
            massive_baselines = json.load(f)
            safe_samples.extend(massive_baselines)
            print(f"    - Loaded {len(massive_baselines)} enterprise safe tools from massive_safe_baselines.json")

    # Poisoned / Shadow Attacks (Label 1)
    if os.path.exists("poisoned_tests.json"):
        with open("poisoned_tests.json", "r") as f:
            poisoned_tests = json.load(f)
            poisoned_samples.extend(poisoned_tests)
            print(f"    - Loaded {len(poisoned_tests)} attack tools from poisoned_tests.json")

    if os.path.exists("secbench_shadow_tests.json"):
        with open("secbench_shadow_tests.json", "r") as f:
            secbench_tests = json.load(f)
            poisoned_samples.extend(secbench_tests)
            print(f"    - Loaded {len(secbench_tests)} shadow attack tools from secbench_shadow_tests.json")

    if not safe_samples or not poisoned_samples:
        raise FileNotFoundError(
            f"Error: Missing dataset files. Please ensure safe_baselines.json and poisoned_tests.json are in the working directory."
        )

    # E. Data Augmentation: Inject diverse, naturally-phrased safe tool descriptions
    # to eliminate training bias toward rigid synthetic patterns.
    print("\n[*] Augmenting safe baselines with diverse real-world tool descriptions...")
    augmented_safe = _generate_diverse_safe_augmentation()
    safe_samples.extend(augmented_safe)
    print(f"    - Injected {len(augmented_safe)} diverse augmentation samples into safe class.")

    print(f"[*] Total ingested data: {len(safe_samples)} Safe (0), {len(poisoned_samples)} Poisoned (1).")

    # C. Embedding Phase
    print("\n[*] Initializing SentenceTransformer('all-MiniLM-L6-v2') for feature extraction...")
    embedder = SentenceTransformer("all-MiniLM-L6-v2", device=device)

    print("[*] Encoding tool schemas into 384-dimensional semantic space...")
    safe_texts = [serialize_tool(t) for t in safe_samples]
    poisoned_texts = [serialize_tool(t) for t in poisoned_samples]

    start_embed = time.time()
    safe_embeddings = embedder.encode(safe_texts, show_progress_bar=True, convert_to_numpy=True)
    poisoned_embeddings = embedder.encode(poisoned_texts, show_progress_bar=True, convert_to_numpy=True)
    embed_time = time.time() - start_embed
    print(f"[*] Completed text vectorization of {len(safe_embeddings) + len(poisoned_embeddings)} items in {embed_time:.2f}s.")

    # Consolidate features and targets
    X = np.vstack([safe_embeddings, poisoned_embeddings]).astype(np.float32)
    y = np.concatenate([np.zeros(len(safe_embeddings)), np.ones(len(poisoned_embeddings))]).astype(np.float32)

    # D. Dataset Splitting & DataLoader Compilation
    if HAS_SKLEARN:
        print("\n[*] Splitting dataset into train/test splits (80/20 ratio)...")
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    else:
        # Simple manual fallback splitting if sklearn is not installed
        print("\n[*] Sklearn unavailable, performing manual 80/20 train/test splitting...")
        indices = np.arange(len(X))
        np.random.seed(42)
        np.random.shuffle(indices)
        split_idx = int(len(X) * 0.8)
        train_indices, test_indices = indices[:split_idx], indices[split_idx:]
        X_train, X_test = X[train_indices], X[test_indices]
        y_train, y_test = y[train_indices], y[test_indices]

    # Convert to PyTorch Tensors
    X_train_t = torch.tensor(X_train)
    y_train_t = torch.tensor(y_train).unsqueeze(1)
    X_test_t = torch.tensor(X_test)
    y_test_t = torch.tensor(y_test).unsqueeze(1)

    train_loader = DataLoader(
        TensorDataset(X_train_t, y_train_t),
        batch_size=32,
        shuffle=True
    )

    # E. PyTorch Model Training Loop
    print("\n[*] Initializing PyTorch MLP training environment...")
    mlp = ToolMLPClassifier(input_dim=384, hidden_dim=64).to(device)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(mlp.parameters(), lr=0.001, weight_decay=1e-5)

    epochs = 30
    print(f"[*] Training for {epochs} epochs...")
    mlp.train()
    for epoch in range(1, epochs + 1):
        running_loss = 0.0
        for batch_x, batch_y in train_loader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            optimizer.zero_grad()
            logits = mlp(batch_x)
            loss = criterion(logits, batch_y)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * batch_x.size(0)

        epoch_loss = running_loss / len(X_train)
        if epoch % 5 == 0 or epoch == 1:
            print(f"    Epoch {epoch:02d}/{epochs:02d} | Avg Loss: {epoch_loss:.5f}")

    # F. Model Evaluation & Metric Output
    print("\n[*] Running validation evaluation on test split...")
    mlp.eval()
    with torch.no_grad():
        test_inputs = X_test_t.to(device)
        test_logits = mlp(test_inputs)
        probs = torch.sigmoid(test_logits).cpu().numpy()
        preds = (probs >= 0.5).astype(np.float32)

    # Calculate metrics
    y_test_np = y_test
    tp = np.sum((preds == 1) & (y_test_np[:, None] == 1))
    fp = np.sum((preds == 1) & (y_test_np[:, None] == 0))
    fn = np.sum((preds == 0) & (y_test_np[:, None] == 1))
    tn = np.sum((preds == 0) & (y_test_np[:, None] == 0))

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
    accuracy = (tp + tn) / len(y_test_np)

    print("\n" + "=" * 50)
    print("            PYTORCH MLP CLASSIFIER METRICS          ")
    print("=" * 50)
    print(f"  - Test Set Accuracy: {accuracy*100:6.2f}%")
    print(f"  - Precision        : {precision:8.4f}")
    print(f"  - Recall           : {recall:8.4f}")
    print(f"  - F1-Score         : {f1:8.4f}")
    print(f"  - Confusion Matrix : TP={tp}, FP={fp}, FN={fn}, TN={tn}")
    print("=" * 50)

    # G. Optional Scikit-Learn Classifiers Comparison
    if HAS_SKLEARN:
        print("\n[*] Training comparison scikit-learn models...")
        # 1. Support Vector Classifier (SVC)
        svc = SVC(probability=True, random_state=42)
        svc.fit(X_train, y_train)
        svc_preds = svc.predict(X_test)
        svc_metrics = precision_recall_fscore_support(y_test, svc_preds, average="binary")
        
        print("\n" + "=" * 50)
        print("          SCIKIT-LEARN SVC CLASSIFIER METRICS       ")
        print("=" * 50)
        print(f"  - Precision        : {svc_metrics[0]:8.4f}")
        print(f"  - Recall           : {svc_metrics[1]:8.4f}")
        print(f"  - F1-Score         : {svc_metrics[2]:8.4f}")
        print("=" * 50)

    # H. Serialization & Disk Export
    model_path = "shield_model.pt"
    print(f"\n[*] Exporting trained PyTorch MLP weights to disk: '{model_path}'...")
    
    # Save both model weights state_dict AND hyperparameters for clean dynamic instantiation
    torch.save({
        'model_state_dict': mlp.state_dict(),
        'input_dim': 384,
        'hidden_dim': 64,
        'threshold': 0.5
    }, model_path)
    
    print("[*] Neural Shield training completed successfully!")
    print("=" * 80)

if __name__ == "__main__":
    main()
