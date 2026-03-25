module.exports = {
  apps: [
    {
      name: "ollama",
      script: "/opt/voiceserverV2/start-ollama.sh",
      autorestart: true,
      error_file: "/var/log/voiceserver/ollama-error.log",
      out_file: "/var/log/voiceserver/ollama.log",
      merge_logs: true,
      time: true,
    },
    {
      name: "voiceserver",
      script: "dist/index.js",
      cwd: "/opt/voiceserverV2",
      env: {
        NODE_ENV: "production",
        HF_HOME: "/root/.cache/huggingface",
      },
      instances: 1,
      exec_mode: "fork",
      autorestart: true,
      watch: false,
      max_memory_restart: "3G",
      error_file: "/var/log/voiceserver/error.log",
      out_file: "/var/log/voiceserver/out.log",
      merge_logs: true,
      time: true,
    },
    // Tunnels removed 2026-03-25 — using direct Caddy reverse proxy + TLS instead
  ],
};
