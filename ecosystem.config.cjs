module.exports = {
  apps: [{
    name: "atobot",
    script: "npx",
    args: "tsx server/index.ts",
    cwd: "/workspaces/atobot-trading",
    exec_mode: "fork",
    instances: 1,
    autorestart: true,
    watch: false,
    max_memory_restart: "500M",
    error_file: "logs/err.log",
    out_file: "logs/out.log",
    merge_logs: true,
    env: {
      NODE_ENV: "development",
      PATH: process.env.PATH
    },
    // Restart delay to prevent rapid crash loops
    restart_delay: 3000,
    // Max restarts within 1 minute before giving up
    max_restarts: 10,
    min_uptime: "10s"
  }, {
    name: "keepalive",
    script: "scripts/keepalive.sh",
    cwd: "/workspaces/atobot-trading",
    interpreter: "/bin/bash",
    autorestart: true,
    watch: false,
    error_file: "logs/keepalive-err.log",
    out_file: "logs/keepalive-out.log",
    merge_logs: true,
    restart_delay: 5000,
    max_restarts: 5,
    min_uptime: "10s"
  }]
};
