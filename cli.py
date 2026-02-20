"""
Mengram CLI

Usage:
    mengram init              # Interactive setup
    mengram init --provider anthropic --api-key sk-ant-...
    mengram server            # Start MCP server
    mengram server --config ~/.mengram/config.yaml
    mengram status            # Check setup
    mengram stats             # Vault statistics
"""

import os
import sys
import json
import yaml
import shutil
import platform
import argparse
from pathlib import Path


# Default paths
DEFAULT_HOME = Path.home() / ".mengram"
DEFAULT_CONFIG = DEFAULT_HOME / "config.yaml"
DEFAULT_VAULT = DEFAULT_HOME / "vault"


def get_claude_desktop_config_path() -> Path:
    """Path to Claude Desktop MCP config"""
    system = platform.system()
    if system == "Darwin":  # macOS
        return Path.home() / "Library" / "Application Support" / "Claude" / "claude_desktop_config.json"
    elif system == "Windows":
        return Path(os.environ.get("APPDATA", "")) / "Claude" / "claude_desktop_config.json"
    else:  # Linux
        return Path.home() / ".config" / "Claude" / "claude_desktop_config.json"


def cmd_init(args):
    """Interactive setup — creates config, vault, MCP integration"""
    print("🧠 Mengram Setup\n")

    home_dir = Path(args.home) if args.home else DEFAULT_HOME
    home_dir.mkdir(parents=True, exist_ok=True)

    config_path = home_dir / "config.yaml"
    vault_path = home_dir / "vault"

    # --- 1. LLM Provider ---
    provider = args.provider
    api_key = args.api_key

    if not provider:
        print("Which LLM provider?")
        print("  1) anthropic  (Claude — recommended)")
        print("  2) openai     (GPT)")
        print("  3) ollama     (local, free)")
        choice = input("\nChoice [1]: ").strip() or "1"
        provider = {"1": "anthropic", "2": "openai", "3": "ollama"}.get(choice, "anthropic")

    if not api_key and provider in ("anthropic", "openai"):
        env_var = "ANTHROPIC_API_KEY" if provider == "anthropic" else "OPENAI_API_KEY"
        env_key = os.environ.get(env_var, "")

        if env_key:
            print(f"\n✅ Found {env_var} in environment")
            api_key = env_key
        else:
            api_key = input(f"\n🔑 Enter your API key: ").strip()
            if not api_key:
                print("❌ API key required. Set it later in config.yaml")
                api_key = "YOUR_API_KEY_HERE"

    # --- 2. Vault path ---
    if args.vault:
        vault_path = Path(args.vault)
    else:
        default_display = str(vault_path)
        custom = input(f"\n📁 Vault path [{default_display}]: ").strip()
        if custom:
            vault_path = Path(custom)

    vault_path.mkdir(parents=True, exist_ok=True)

    # --- 3. Write config ---
    config = {
        "vault_path": str(vault_path),
        "llm": {
            "provider": provider,
        },
        "semantic_search": {
            "enabled": True,
        },
    }

    if provider == "anthropic":
        config["llm"]["anthropic"] = {
            "api_key": api_key,
            "model": "claude-sonnet-4-20250514",
        }
    elif provider == "openai":
        config["llm"]["openai"] = {
            "api_key": api_key,
            "model": "gpt-4o-mini",
        }
    elif provider == "ollama":
        config["llm"]["ollama"] = {
            "base_url": "http://localhost:11434",
            "model": "llama3.2",
        }

    with open(config_path, "w") as f:
        yaml.dump(config, f, default_flow_style=False, allow_unicode=True)
    print(f"\n✅ Config: {config_path}")

    # --- 4. Create run_server.sh ---
    server_script = home_dir / "run_server.sh"
    python_path = sys.executable

    script_content = f"""#!/bin/bash
cd "{home_dir}"
"{python_path}" -m api.mcp_server "{config_path}"
"""
    with open(server_script, "w") as f:
        f.write(script_content)
    os.chmod(server_script, 0o755)
    print(f"✅ Server script: {server_script}")

    # --- 5. Find package location for MCP ---
    try:
        import mengram
        package_dir = Path(mengram.__file__).parent
        # If installed as package, the engine/ and api/ are siblings
        # We need the directory containing api/mcp_server.py
        if (package_dir / "api" / "mcp_server.py").exists():
            mcp_working_dir = str(package_dir)
        elif (package_dir.parent / "api" / "mcp_server.py").exists():
            mcp_working_dir = str(package_dir.parent)
        else:
            mcp_working_dir = str(home_dir)
    except ImportError:
        mcp_working_dir = str(home_dir)

    # --- 6. Claude Desktop MCP integration ---
    claude_config_path = get_claude_desktop_config_path()
    setup_mcp = True

    if not args.no_mcp:
        if not claude_config_path.parent.exists():
            print(f"\n⚠️  Claude Desktop config dir not found: {claude_config_path.parent}")
            print("   Install Claude Desktop first, then run: mengram init --mcp-only")
            setup_mcp = False
        else:
            # Read existing config
            claude_config = {}
            if claude_config_path.exists():
                try:
                    with open(claude_config_path) as f:
                        claude_config = json.load(f)
                except (json.JSONDecodeError, Exception):
                    claude_config = {}

            # Add MCP server
            if "mcpServers" not in claude_config:
                claude_config["mcpServers"] = {}

            claude_config["mcpServers"]["mengram"] = {
                "command": str(server_script),
            }

            with open(claude_config_path, "w") as f:
                json.dump(claude_config, f, indent=2)
            print(f"✅ Claude Desktop MCP: {claude_config_path}")
    else:
        setup_mcp = False

    # --- Done ---
    print(f"\n{'='*50}")
    print(f"🎉 Mengram ready!\n")
    print(f"   Config:  {config_path}")
    print(f"   Vault:   {vault_path}")
    print(f"   LLM:     {provider}")
    print(f"   Search:  semantic (local embeddings)")

    if setup_mcp:
        print(f"\n   ⚡ Restart Claude Desktop to activate MCP")
        print(f"   Then tell Claude: 'Remember that I work at ...'")
    else:
        print(f"\n   Start MCP server: mengram server")

    print(f"\n   Python SDK:")
    print(f"   >>> from mengram import Memory")
    print(f"   >>> m = Memory(vault_path='{vault_path}', llm_provider='{provider}')")


def cmd_server(args):
    """Start MCP server"""
    if getattr(args, 'cloud', False):
        # Cloud mode — connect to cloud API
        api_key = os.environ.get("MENGRAM_API_KEY", "")
        base_url = os.environ.get("MENGRAM_URL", "https://mengram.io")
        user_id = os.environ.get("MENGRAM_USER_ID", "default")

        if not api_key:
            print("❌ Set MENGRAM_API_KEY environment variable")
            print("   Get one: curl -X POST https://mengram.io/v1/signup -d '{\"email\": \"you@email.com\"}'")
            sys.exit(1)

        print(f"🧠 Starting Mengram Cloud MCP server...", file=sys.stderr)
        print(f"   API: {base_url}", file=sys.stderr)

        import asyncio
        from api.cloud_mcp_server import main as cloud_mcp_main
        asyncio.run(cloud_mcp_main())
        return

    config_path = args.config or str(DEFAULT_CONFIG)

    if not Path(config_path).exists():
        print(f"❌ Config not found: {config_path}")
        print(f"   Run: mengram init")
        sys.exit(1)

    print(f"🧠 Starting Mengram MCP server...")
    print(f"   Config: {config_path}")

    # Set working directory to where engine/ is
    try:
        import engine
        engine_dir = Path(engine.__file__).parent.parent
        os.chdir(engine_dir)
    except ImportError:
        pass

    import asyncio
    from api.mcp_server import main as mcp_main
    # Monkey-patch sys.argv for mcp_server
    sys.argv = ["mcp_server", config_path]
    asyncio.run(mcp_main())


def cmd_status(args):
    """Check setup status"""
    print("🧠 Mengram Status\n")

    # Config
    config_path = DEFAULT_CONFIG
    if config_path.exists():
        print(f"✅ Config: {config_path}")
        with open(config_path) as f:
            config = yaml.safe_load(f)
        print(f"   Provider: {config.get('llm', {}).get('provider', '?')}")
        print(f"   Vault: {config.get('vault_path', '?')}")
    else:
        print(f"❌ Config not found. Run: mengram init")
        return

    # Vault
    vault_path = Path(config.get("vault_path", ""))
    if vault_path.exists():
        notes = list(vault_path.glob("*.md"))
        print(f"✅ Vault: {len(notes)} notes")
    else:
        print(f"⚠️  Vault empty")

    # Vector DB
    vectors_db = vault_path / ".vectors.db"
    if vectors_db.exists():
        size = vectors_db.stat().st_size
        print(f"✅ Vector index: {size / 1024:.0f}KB")
    else:
        print(f"⚠️  No vector index yet (will be created on first use)")

    # Claude Desktop
    claude_config = get_claude_desktop_config_path()
    if claude_config.exists():
        try:
            with open(claude_config) as f:
                cc = json.load(f)
            if "mengram" in cc.get("mcpServers", {}):
                print(f"✅ Claude Desktop MCP configured")
            else:
                print(f"⚠️  Claude Desktop found but MCP not configured")
        except Exception:
            print(f"⚠️  Claude Desktop config error")
    else:
        print(f"⚠️  Claude Desktop not found")

    # sentence-transformers
    try:
        import sentence_transformers
        print(f"✅ sentence-transformers installed")
    except ImportError:
        print(f"⚠️  sentence-transformers not installed: pip install sentence-transformers")


def cmd_stats(args):
    """Show vault statistics"""
    config_path = args.config or str(DEFAULT_CONFIG)

    if not Path(config_path).exists():
        print(f"❌ Run: mengram init")
        sys.exit(1)

    from engine.brain import create_brain
    # Monkey-patch for config path
    old_argv = sys.argv
    sys.argv = ["", config_path]

    brain = create_brain(config_path)
    stats = brain.get_stats()

    print("🧠 Mengram Stats\n")
    vault = stats.get("vault", {})
    print(f"📁 Notes: {vault.get('total_notes', 0)}")
    for t, count in vault.get("by_type", {}).items():
        print(f"   {t}: {count}")

    if "vectors" in stats:
        v = stats["vectors"]
        print(f"\n🔍 Vector Index: {v.get('total_chunks', 0)} chunks, {v.get('total_entities', 0)} entities")

    sys.argv = old_argv


def cmd_api(args):
    """Start REST API server"""
    config_path = args.config or str(DEFAULT_CONFIG)

    if not Path(config_path).exists():
        print(f"❌ Run: mengram init")
        sys.exit(1)

    try:
        import fastapi
        import uvicorn
    except ImportError:
        print("❌ FastAPI not installed: pip install mengram[api]")
        sys.exit(1)

    from engine.brain import create_brain
    from api.rest_server import create_rest_api

    brain = create_brain(config_path)

    # Warmup vector store
    if brain.use_vectors:
        _ = brain.vector_store

    app = create_rest_api(brain)

    print(f"🧠 Mengram REST API")
    print(f"   http://localhost:{args.port}")
    print(f"   Docs: http://localhost:{args.port}/docs")

    uvicorn.run(app, host=args.host, port=args.port, log_level="info")


def cmd_import(args):
    """Import existing data into memory"""
    import_type = args.import_type
    if not import_type:
        print("Usage: mengram import {chatgpt,obsidian,files} <path>")
        print("  mengram import chatgpt ~/Downloads/chatgpt-export.zip")
        print("  mengram import obsidian ~/Documents/MyVault")
        print("  mengram import files notes/*.md")
        sys.exit(1)

    from importer import (
        import_chatgpt, import_obsidian, import_files, RateLimiter,
    )

    # --- Resolve add_fn ---
    if getattr(args, "cloud", False):
        api_key = os.environ.get("MENGRAM_API_KEY", "")
        if not api_key:
            print("❌ Set MENGRAM_API_KEY environment variable")
            sys.exit(1)

        from cloud.client import CloudMemory
        mem = CloudMemory(api_key=api_key)
        limiter = RateLimiter(max_per_minute=100)

        def add_fn(messages):
            limiter.wait_if_needed()
            return mem.add(messages)

        print("☁️  Importing to cloud memory...")
    else:
        config_path = str(DEFAULT_CONFIG)
        if not Path(config_path).exists():
            print("❌ Run: mengram init  (or use --cloud for cloud API)")
            sys.exit(1)

        from engine.brain import create_brain
        brain = create_brain(config_path)
        add_fn = brain.remember
        print("💾 Importing to local memory...")

    # --- Progress callback ---
    def on_progress(current, total, title):
        pct = int(current / total * 100) if total else 0
        bar = "█" * (pct // 5) + "░" * (20 - pct // 5)
        print(f"\r  {bar} {pct}% ({current}/{total}) {title[:40]}", end="", flush=True)

    # --- Run importer ---
    print()
    if import_type == "chatgpt":
        result = import_chatgpt(args.path, add_fn,
                                chunk_size=args.chunk_size, on_progress=on_progress)
    elif import_type == "obsidian":
        result = import_obsidian(args.path, add_fn,
                                 chunk_chars=args.chunk_chars, on_progress=on_progress)
    elif import_type == "files":
        result = import_files(args.paths, add_fn,
                              chunk_chars=args.chunk_chars, on_progress=on_progress)
    else:
        print(f"❌ Unknown import type: {import_type}")
        sys.exit(1)

    # --- Summary ---
    print(f"\n\n{'='*50}")
    print(f"✅ Import complete!\n")
    print(f"   Found:    {result.conversations_found} {'conversations' if import_type == 'chatgpt' else 'files'}")
    print(f"   Imported: {result.chunks_sent} chunks")
    print(f"   Entities: {len(result.entities_created)}")
    print(f"   Time:     {result.duration_seconds:.1f}s")
    if result.errors:
        print(f"\n   ⚠️  {len(result.errors)} errors:")
        for err in result.errors[:5]:
            print(f"      - {err}")
        if len(result.errors) > 5:
            print(f"      ... and {len(result.errors) - 5} more")


def cmd_web(args):
    """Start Web UI — chat + knowledge graph"""
    config_path = args.config or str(DEFAULT_CONFIG)

    if not Path(config_path).exists():
        print(f"❌ Run: mengram init")
        sys.exit(1)

    try:
        import fastapi
        import uvicorn
    except ImportError:
        print("❌ FastAPI not installed: pip install mengram[api]")
        sys.exit(1)

    from engine.brain import create_brain
    from api.rest_server import create_rest_api

    brain = create_brain(config_path)

    if brain.use_vectors:
        _ = brain.vector_store

    app = create_rest_api(brain)

    url = f"http://localhost:{args.port}"
    print(f"🧠 Mengram Web UI")
    print(f"   {url}")
    print(f"   API docs: {url}/docs")
    print()

    if not args.no_open:
        import threading
        import webbrowser
        threading.Timer(1.0, lambda: webbrowser.open(url)).start()

    uvicorn.run(app, host="0.0.0.0", port=args.port, log_level="info")


def main():
    parser = argparse.ArgumentParser(
        prog="mengram",
        description="🧠 Mengram — AI memory layer for apps",
    )
    sub = parser.add_subparsers(dest="command")

    # init
    p_init = sub.add_parser("init", help="Setup Mengram")
    p_init.add_argument("--provider", choices=["anthropic", "openai", "ollama"], help="LLM provider")
    p_init.add_argument("--api-key", help="API key")
    p_init.add_argument("--vault", help="Custom vault path")
    p_init.add_argument("--home", help="Mengram home dir (default: ~/.mengram)")
    p_init.add_argument("--no-mcp", action="store_true", help="Skip Claude Desktop MCP setup")
    p_init.add_argument("--mcp-only", action="store_true", help="Only setup MCP (config must exist)")

    # server
    p_server = sub.add_parser("server", help="Start MCP server")
    p_server.add_argument("--config", help="Config path (default: ~/.mengram/config.yaml)")
    p_server.add_argument("--cloud", action="store_true", help="Use cloud API instead of local vault")

    # status
    sub.add_parser("status", help="Check setup status")

    # stats
    p_stats = sub.add_parser("stats", help="Vault statistics")
    p_stats.add_argument("--config", help="Config path")

    # api
    p_api = sub.add_parser("api", help="Start REST API server")
    p_api.add_argument("--config", help="Config path")
    p_api.add_argument("--host", default="0.0.0.0", help="Host (default: 0.0.0.0)")
    p_api.add_argument("--port", type=int, default=8420, help="Port (default: 8420)")

    # import
    p_import = sub.add_parser("import", help="Import existing data into memory")
    import_sub = p_import.add_subparsers(dest="import_type")

    p_chatgpt = import_sub.add_parser("chatgpt", help="Import ChatGPT export ZIP")
    p_chatgpt.add_argument("path", help="Path to ChatGPT export ZIP file")
    p_chatgpt.add_argument("--chunk-size", type=int, default=20, dest="chunk_size")
    p_chatgpt.add_argument("--cloud", action="store_true", help="Use cloud API")

    p_obsidian = import_sub.add_parser("obsidian", help="Import Obsidian vault")
    p_obsidian.add_argument("path", help="Path to Obsidian vault directory")
    p_obsidian.add_argument("--chunk-chars", type=int, default=4000, dest="chunk_chars")
    p_obsidian.add_argument("--cloud", action="store_true", help="Use cloud API")

    p_files = import_sub.add_parser("files", help="Import text/markdown files")
    p_files.add_argument("paths", nargs="+", help="File paths")
    p_files.add_argument("--chunk-chars", type=int, default=4000, dest="chunk_chars")
    p_files.add_argument("--cloud", action="store_true", help="Use cloud API")

    # web
    p_web = sub.add_parser("web", help="Start Web UI (chat + knowledge graph)")
    p_web.add_argument("--config", help="Config path")
    p_web.add_argument("--port", type=int, default=8420, help="Port (default: 8420)")
    p_web.add_argument("--no-open", action="store_true", help="Don't open browser")

    args = parser.parse_args()

    if args.command == "init":
        cmd_init(args)
    elif args.command == "server":
        cmd_server(args)
    elif args.command == "status":
        cmd_status(args)
    elif args.command == "stats":
        cmd_stats(args)
    elif args.command == "api":
        cmd_api(args)
    elif args.command == "import":
        cmd_import(args)
    elif args.command == "web":
        cmd_web(args)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
